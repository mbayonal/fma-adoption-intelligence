"""Pipeline ETL + scoring para FMA Adoption Intelligence (entrega_3).

Carga los CSVs fuente, construye el feature store, aplica el modelo RF
pre-entrenado, calcula SHAP y agrega scores por tenant.  No re-entrena.

Uso:
    python pipeline.py

Salidas en data/:
    analytical_dataset.parquet   — feature store + score + risk_level por técnico
    tenant_scores.parquet        — score agregado + métricas por tenant
    shap_values.parquet          — top-3 factores SHAP por técnico
    feature_importance.parquet   — importancia global (mean |SHAP|)
    last_updated.txt             — timestamp ISO de la última ejecución
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from src import etl
from src import config as cfg


def _load_metadata() -> dict:
    return json.loads(cfg.RF_METADATA.read_text())


def _score(rf_pipeline, feat: pd.DataFrame, top_num: list[str], top_cat: list[str]) -> pd.Series:
    selected = top_num + top_cat
    X = feat[selected].copy()
    for c in top_num:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in top_cat:
        X[c] = X[c].fillna("Unknown").astype("string")
    return pd.Series(rf_pipeline.predict_proba(X)[:, 1], index=feat.index)


def _shap_analysis(rf_pipeline, feat: pd.DataFrame, top_num: list[str], top_cat: list[str]):
    selected = top_num + top_cat
    X = feat[selected].copy()
    for c in top_num:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    for c in top_cat:
        X[c] = X[c].fillna("Unknown").astype("string")

    pre = rf_pipeline.named_steps["pre"]
    clf = rf_pipeline.named_steps["clf"]
    X_tr = pre.transform(X)
    if hasattr(X_tr, "toarray"):
        X_tr = X_tr.toarray()

    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        feat_names = [f"f{i}" for i in range(X_tr.shape[1])]

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_tr)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    rows = []
    for i in range(sv.shape[0]):
        order = np.argsort(np.abs(sv[i]))[::-1][:3]
        rows.append(
            {
                "TECHNICIAN_ID": feat["TECHNICIAN_ID"].iloc[i],
                "pred_score": float(feat["pred_score"].iloc[i]),
                "factor_1": feat_names[order[0]],
                "shap_1": float(sv[i, order[0]]),
                "factor_2": feat_names[order[1]],
                "shap_2": float(sv[i, order[1]]),
                "factor_3": feat_names[order[2]],
                "shap_3": float(sv[i, order[2]]),
            }
        )
    shap_df = pd.DataFrame(rows)

    mean_abs = np.abs(sv).mean(axis=0)
    feat_imp = (
        pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return shap_df, feat_imp


def _aggregate_tenants(feat: pd.DataFrame) -> pd.DataFrame:
    if "TENANT_ID" not in feat.columns:
        return pd.DataFrame()

    tenant_cols = [c for c in ["TENANT_PRIMARY_INDUSTRY", "TENANT_SIZE", "OPS_SIZE_SEGMENT",
                                "FMA_MIGRATION_STATUS_30D", "fma_penetration_30d",
                                "CORE_HEALTH_SCORE"] if c in feat.columns]
    first_tenant_info = feat.drop_duplicates("TENANT_ID")[["TENANT_ID"] + tenant_cols]

    agg = (
        feat.groupby("TENANT_ID")
        .agg(
            n_techs=("TECHNICIAN_ID", "count"),
            tenant_score_mean=("pred_score", "mean"),
            tenant_score_max=("pred_score", "max"),
            tenant_pct_high_risk=("pred_score", lambda x: float((x < 0.5).mean())),
            tenant_pct_critical_risk=("pred_score", lambda x: float((x < 0.3).mean())),
            n_techs_critical=("pred_score", lambda x: int((x < 0.3).sum())),
            n_techs_high=("pred_score", lambda x: int(((x >= 0.3) & (x < 0.5)).sum())),
            retention_real=("label_retained", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(first_tenant_info, on="TENANT_ID", how="left")

    bins = [0, 0.3, 0.5, 0.7, 1.0]
    labels = ["Crítico", "Alto", "Medio", "Bajo"]
    agg["tenant_risk_level"] = pd.cut(agg["tenant_score_mean"], bins=bins, labels=labels)

    if len(agg) >= 4:
        try:
            agg["score_quartile"] = pd.qcut(
                agg["tenant_score_mean"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
            )
        except Exception:
            agg["score_quartile"] = pd.NA

    return agg.reset_index(drop=True)


def main() -> None:
    t0 = time.perf_counter()
    meta = _load_metadata()
    top_num: list[str] = meta["top_num"]
    top_cat: list[str] = meta["top_cat"]

    print("[1/4] ETL — cargando y construyendo feature store...")
    raw_t, raw_te, raw_e = etl.load_raw()
    clean = etl.run_phase1(raw_t, raw_te, raw_e)
    feat = etl.build_feature_store(clean)
    print(f"      n={len(feat)}  retenidos={int(feat['label_retained'].sum())}  "
          f"abandonos={int((1 - feat['label_retained']).sum())}")

    print("[2/4] Scoring con modelo RF pre-entrenado...")
    rf_pipeline = joblib.load(cfg.RF_MODEL)
    feat["pred_score"] = _score(rf_pipeline, feat, top_num, top_cat)
    bins = [0, 0.3, 0.5, 0.7, 1.0]
    labels_risk = ["Crítico", "Alto", "Medio", "Bajo"]
    feat["risk_level"] = pd.cut(feat["pred_score"], bins=bins, labels=labels_risk)
    print(f"      score_mean={feat['pred_score'].mean():.3f}  "
          f"crítico={int((feat['pred_score'] < 0.3).sum())}  "
          f"alto={int(((feat['pred_score'] >= 0.3) & (feat['pred_score'] < 0.5)).sum())}")

    print("[3/4] SHAP — factores dominantes por técnico...")
    shap_df, feat_imp = _shap_analysis(rf_pipeline, feat, top_num, top_cat)

    print("[4/4] Agregación por tenant + guardando Parquet...")
    tenant_scores = _aggregate_tenants(feat)

    feat.to_parquet(cfg.OUT_DIR / "analytical_dataset.parquet", index=False)
    shap_df.to_parquet(cfg.OUT_DIR / "shap_values.parquet", index=False)
    feat_imp.to_parquet(cfg.OUT_DIR / "feature_importance.parquet", index=False)
    if not tenant_scores.empty:
        tenant_scores.to_parquet(cfg.OUT_DIR / "tenant_scores.parquet", index=False)
        print(f"      tenants={len(tenant_scores)}  "
              f"críticos={int((tenant_scores['tenant_risk_level'] == 'Crítico').sum())}  "
              f"altos={int((tenant_scores['tenant_risk_level'] == 'Alto').sum())}")

    (cfg.DATA_DIR / "last_updated.txt").write_text(pd.Timestamp.now().isoformat())
    print(f"\nPipeline completado en {time.perf_counter() - t0:.1f}s.")


if __name__ == "__main__":
    main()
