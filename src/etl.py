"""ETL: limpieza por entidad y construcción del feature store por técnico.

Implementa el pipeline definido en entrega_1/docs/entregables_manu/07_etl.md:
filtros de población, ventanas T0/T1, label binaria, bloques A/B/C/D.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg


@dataclass
class CleanData:
    tenants: pd.DataFrame
    technicians: pd.DataFrame
    events: pd.DataFrame
    device_by_technician: pd.DataFrame


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _canonical_role(role: str | float) -> str:
    if not isinstance(role, str) or not role.strip():
        return "Unknown"
    r = role.lower()
    rules = [
        (r"sales|ventas|csr|comfort advisor", "Sales"),
        (r"helper|apprentice|aprendiz", "Helper"),
        (r"install", "Install"),
        (r"plumb", "Plumbing"),
        (r"electric", "Electrical"),
        (r"hvac", "HVAC"),
        (r"service", "Service"),
        (r"admin|office|dispatch", "Admin"),
        (r"technician|tech\b", "Technician"),
    ]
    for pattern, label in rules:
        if re.search(pattern, r):
            return label
    return "Other"


def clean_tenants(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df = df[df["TENANT_STATUS"].isin(cfg.VALID_TENANT_STATUSES)].copy()

    df["SUPPORT_CASES_OPEN"] = df["SUPPORT_CASES_OPEN"].clip(lower=0)
    df["nps_is_missing"] = df["TENANT_AVG_NPS_SCORE"].isna()

    fma = df.get("TECHNICIANS_USING_FMA_30D", pd.Series(0, index=df.index))
    mob = df.get("TECHNICIANS_USING_MOBILE_30D", pd.Series(0, index=df.index))
    df["fma_penetration_30d"] = np.where(mob > 0, fma / mob, 0.0)

    return df.reset_index(drop=True)


def clean_technicians(df_raw: pd.DataFrame, valid_tenant_ids: set) -> pd.DataFrame:
    df = df_raw.copy()
    mask = (
        df["IS_ACTIVE_TECHNICIAN"].astype(bool)
        & df["TECHNICIAN_DEVICE_DATA_SOURCE"].isin(cfg.VALID_DEVICE_SOURCES)
        & df["TENANT_ID"].isin(valid_tenant_ids)
    )
    df = df[mask].copy()
    df["role_canonical"] = df["TECHNICIAN_ROLE"].apply(_canonical_role)
    df["has_team_assigned"] = df["TECHNICIAN_TEAM"].notna()
    drop_cols = [c for c in df.columns if c.startswith("TECHNICIAN_DEVICE_")]
    df = df.drop(columns=drop_cols)
    return df.reset_index(drop=True)


def clean_events(df_raw: pd.DataFrame, valid_technician_ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_raw.copy()
    df = df[df["TECHNICIAN_ID"].isin(valid_technician_ids)].copy()
    df["feature_name_clean"] = (
        df["FEATURE_NAME"].astype("string").str.replace(r"\s*\*$", "", regex=True).fillna("Unknown")
    )
    df["EVENT_DATE"] = _parse_date(df["EVENT_DATE"])
    df["EVENT_TIME_PARSED"] = _parse_date(df["EVENT_TIME"])
    df["event_time_is_valid"] = df["EVENT_TIME_PARSED"].notna()

    sess_size = df.groupby("SESSION_ID")["SESSION_ID"].transform("size")
    df["events_in_session"] = sess_size
    df["is_anomalous_session"] = sess_size > 500

    device_cols = ["DEVICE_OPERATING_SYSTEM", "DEVICE_BRAND", "APP_VERSION", "DEVICE_MODEL"]

    def _mode(s: pd.Series):
        m = s.dropna().mode()
        return m.iloc[0] if not m.empty else np.nan

    device = df.groupby("TECHNICIAN_ID")[device_cols].agg(_mode).reset_index()
    device = device.rename(columns={c: f"{c.lower()}_mode" for c in device_cols})

    app_max = (
        df.dropna(subset=["APP_VERSION"]).groupby("TECHNICIAN_ID")["APP_VERSION"].max().reset_index()
    )
    app_max = app_max.rename(columns={"APP_VERSION": "app_version_max"})
    device = device.merge(app_max, on="TECHNICIAN_ID", how="left")

    return df.reset_index(drop=True), device


def run_phase1(
    raw_tenants: pd.DataFrame, raw_techs: pd.DataFrame, raw_events: pd.DataFrame
) -> CleanData:
    tenants = clean_tenants(raw_tenants)
    technicians = clean_technicians(raw_techs, set(tenants["TENANT_ID"]))
    events, device_by_tech = clean_events(raw_events, set(technicians["TECHNICIAN_ID"]))
    return CleanData(tenants, technicians, events, device_by_tech)


def _shannon_entropy(counts: pd.Series) -> float:
    p = counts.values.astype(float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def _features_block_a(events_tech: pd.DataFrame, n_days: int) -> dict:
    sub = events_tech[events_tech["days_since_first"].between(0, n_days - 1)]
    n_events = len(sub)
    out = {
        f"obs_events_count_w{n_days}": n_events,
        f"obs_events_count_w{n_days}_log": float(np.log1p(n_events)),
        f"obs_sessions_count_w{n_days}": int(sub["SESSION_ID"].nunique()),
        f"obs_active_days_w{n_days}": int(sub["EVENT_DATE"].nunique()),
        f"obs_active_days_ratio_w{n_days}": float(sub["EVENT_DATE"].nunique() / n_days),
        f"obs_unique_features_w{n_days}": int(sub["feature_name_clean"].nunique()),
        f"obs_feature_entropy_w{n_days}": _shannon_entropy(
            sub["feature_name_clean"].value_counts()
        ),
        f"obs_unique_screens_w{n_days}": int(sub["SCREEN_NAME"].nunique()),
        f"obs_screen_view_ratio_w{n_days}": float(
            (sub["feature_name_clean"] == "Screen View").sum() / n_events if n_events else 0
        ),
    }
    for fname in ["Job Close Out", "Time", "Payment", "Estimate", "Invoice", "Form", "Media"]:
        key = "obs_" + fname.lower().replace(" ", "_") + f"_count_w{n_days}"
        out[key] = int((sub["feature_name_clean"] == fname).sum())
        out[key.replace("_count_", "_pct_")] = float(out[key] / n_events) if n_events else 0.0
    return out


def _features_block_a_growth(events_tech: pd.DataFrame, n_days: int) -> dict:
    """Trayectoria intra-ventana: divide la ventana de observación en dos mitades.

    Para n_days=14: early=[0,6], late=[7,13]. Ambas mitades dentro de T0 (sin leakage).
    Para n_days=28: early=[0,13], late=[14,27]. Idem.
    """
    half = n_days // 2
    early = events_tech[events_tech["days_since_first"].between(0, half - 1)]
    late = events_tech[events_tech["days_since_first"].between(half, n_days - 1)]
    return {
        f"obs_events_early_half_w{n_days}": int(len(early)),
        f"obs_events_late_half_w{n_days}": int(len(late)),
        f"obs_events_growth_ratio_w{n_days}": float((len(late) + 1) / (len(early) + 1)),
        f"obs_active_days_early_half_w{n_days}": int(early["EVENT_DATE"].nunique()),
        f"obs_active_days_late_half_w{n_days}": int(late["EVENT_DATE"].nunique()),
    }


def build_feature_store(clean: CleanData) -> pd.DataFrame:
    """Construye una fila por técnico con bloques A, B, C, D."""

    events = clean.events.dropna(subset=["EVENT_DATE"]).copy()
    first_session = events.groupby("TECHNICIAN_ID")["EVENT_DATE"].min().rename("first_session_date")
    events = events.merge(first_session, on="TECHNICIAN_ID", how="left")
    events["days_since_first"] = (events["EVENT_DATE"] - events["first_session_date"]).dt.days

    obs_n = cfg.OBSERVATION_WINDOW_DAYS
    label_start = obs_n + cfg.LABEL_GAP_DAYS
    label_end = label_start + cfg.LABEL_WINDOW_DAYS - 1
    max_obs_day = events["EVENT_DATE"].max()

    rows = []
    for tech_id, ev in events.groupby("TECHNICIAN_ID"):
        first = ev["first_session_date"].iloc[0]
        days_observable = (max_obs_day - first).days
        if days_observable < obs_n + cfg.MIN_OBSERVABLE_DAYS_AFTER_OBS:
            continue

        block_a = _features_block_a(ev, obs_n)
        block_a.update(_features_block_a_growth(ev, obs_n))

        observable_label_end = min(label_end, days_observable)
        label_sub = ev[ev["days_since_first"].between(label_start, observable_label_end)]
        label_active_days = int(label_sub["EVENT_DATE"].nunique())
        label_window_days = max(0, observable_label_end - label_start + 1)
        label_window_complete = days_observable >= label_end
        label = int(label_active_days >= cfg.LABEL_ACTIVE_DAYS_THRESHOLD)

        rows.append(
            {
                "TECHNICIAN_ID": tech_id,
                "first_session_date": first,
                **block_a,
                "label_active_days": label_active_days,
                "label_window_days_observable": label_window_days,
                "label_window_complete": label_window_complete,
                "label_retained": label,
            }
        )

    feat = pd.DataFrame(rows)

    techs = clean.technicians[
        ["TECHNICIAN_ID", "TENANT_ID", "role_canonical", "has_team_assigned", "IS_MANAGED_TECHNICIAN"]
    ].copy()
    feat = feat.merge(techs, on="TECHNICIAN_ID", how="left")

    feat = feat.merge(clean.device_by_technician, on="TECHNICIAN_ID", how="left")

    tenant_cols = [
        "TENANT_ID",
        "TENANT_SIZE",
        "TENANT_BUSINESS_FOCUS",
        "TENANT_PRIMARY_INDUSTRY",
        "TENANT_TRADE_CLASSIFICATION",
        "OPS_SIZE_SEGMENT",
        "CORE_HEALTH_SCORE",
        "TENANT_AVG_NPS_SCORE",
        "nps_is_missing",
        "TECHNICIAN_LICENCES",
        "fma_penetration_30d",
        "FMA_MIGRATION_STATUS_30D",
    ]
    avail = [c for c in tenant_cols if c in clean.tenants.columns]
    feat = feat.merge(clean.tenants[avail], on="TENANT_ID", how="left")

    if "TECHNICIAN_LICENCES" in feat:
        feat["tenant_technician_licenses_log"] = np.log1p(feat["TECHNICIAN_LICENCES"].fillna(0))

    return feat.reset_index(drop=True)


def load_raw():
    tenants = pd.read_csv(cfg.RAW_TENANTS, low_memory=False)
    techs = pd.read_csv(cfg.RAW_TECHS, low_memory=False)
    events = pd.read_csv(cfg.RAW_EVENTS, low_memory=False)
    return tenants, techs, events
