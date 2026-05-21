"""FMA Adoption Intelligence — Tablero analítico interactivo (Streamlit).

Vistas:
  1. Factores de Adopción  — descriptivo (R2, R14.1)
  2. Puntuación de Riesgo  — predictivo  (R1, R3, R5-R10, R14.2)
  3. Guía de Migración     — prescriptivo (R4, R14.3)

Requiere ejecutar pipeline.py antes de iniciar el tablero.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, ttest_ind

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FMA Adoption Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_DIR = Path(__file__).parent / "data"
AUC_VAL = 0.829

# ── global CSS (mockup design system) ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], section.main > div { background: #F1F5F9 !important; }

*, *::before, *::after { font-family: 'Inter', system-ui, sans-serif; }

/* hide streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── top navbar ── */
.fma-navbar {
  background: #0F172A;
  padding: 14px 32px;
  display: flex;
  align-items: center;
  gap: 24px;
  border-radius: 10px;
  margin-bottom: 6px;
}
.fma-brand {
  font-size: 14px;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.2px;
}
.fma-brand em { color: #F97316; font-style: normal; }
.fma-brand span { color: rgba(255,255,255,0.4); font-weight: 400; }
.fma-divider { width: 1px; height: 18px; background: rgba(255,255,255,0.12); }
.fma-sub { font-size: 12px; color: rgba(255,255,255,0.5); }
.fma-updated { margin-left: auto; font-size: 11px; color: rgba(255,255,255,0.4); }

/* ── section label ── */
.section-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.1px;
  color: #64748B;
  border-bottom: 1px solid #CBD5E1;
  padding-bottom: 8px;
  margin: 20px 0 14px;
}

/* ── KPI cards ── */
.kpi-grid { display: grid; gap: 12px; margin-bottom: 24px; }
.kpi-grid-4 { grid-template-columns: repeat(4,1fr); }
.kpi-grid-5 { grid-template-columns: repeat(5,1fr); }
.kpi-grid-2 { grid-template-columns: repeat(2,1fr); }
.kpi-card {
  background: #fff;
  border: 1px solid #CBD5E1;
  border-radius: 10px;
  padding: 18px 20px;
}
.kpi-label {
  font-size: 11px;
  font-weight: 600;
  color: #64748B;
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}
.kpi-value {
  font-size: 28px;
  font-weight: 700;
  color: #0F172A;
  letter-spacing: -1px;
  line-height: 1;
}
.kpi-sub { font-size: 11px; color: #64748B; margin-top: 4px; }
.kpi-red    .kpi-value { color: #B91C1C; }
.kpi-amber  .kpi-value { color: #B45309; }
.kpi-green  .kpi-value { color: #15803D; }
.kpi-blue   .kpi-value { color: #1D4ED8; }
.kpi-purple .kpi-value { color: #6D28D9; }

/* ── insight box ── */
.insight-box {
  background: #EFF6FF;
  border: 1px solid #BFDBFE;
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 13px;
  color: #1E3A5F;
  margin-bottom: 20px;
  line-height: 1.65;
}
.insight-box strong { color: #1D4ED8; }

/* ── card wrapper ── */
.fma-card {
  background: #fff;
  border: 1px solid #CBD5E1;
  border-radius: 10px;
  padding: 20px 22px;
  margin-bottom: 14px;
}
.card-title {
  font-size: 13px;
  font-weight: 600;
  color: #0F172A;
  margin-bottom: 4px;
}
.card-sub { font-size: 12px; color: #64748B; font-style: italic; }

/* ── risk badge ── */
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}
.badge-critico { background: #FEF2F2; color: #B91C1C; }
.badge-alto    { background: #FFFBEB; color: #B45309; }
.badge-medio   { background: #F0FDF4; color: #15803D; }
.badge-bajo    { background: #EFF6FF; color: #1D4ED8; }

/* ── score bar ── */
.score-wrap { display: flex; align-items: center; gap: 10px; }
.score-track {
  flex: 1; height: 6px; background: #E2E8F0;
  border-radius: 3px; overflow: hidden; min-width: 80px;
}
.score-fill { height: 100%; border-radius: 3px; }
.score-num { font-size: 12px; font-weight: 600; color: #334155; width: 36px; text-align: right; }

/* ── comparison table ── */
.comp-table { width: 100%; border-collapse: collapse; }
.comp-table thead th {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748B;
  padding: 8px 10px;
  border-bottom: 1px solid #CBD5E1;
  text-align: left;
}
.comp-table th.th-green { color: #15803D; }
.comp-table th.th-red   { color: #B91C1C; }
.comp-table tbody td {
  padding: 9px 10px;
  font-size: 13px;
  border-bottom: 1px solid #F8FAFC;
  color: #0F172A;
  vertical-align: middle;
}
.comp-table tbody tr:hover td { background: #F8FAFC; }
.comp-table tbody tr:last-child td { border-bottom: none; }
.val-pos { font-weight: 700; color: #15803D; }
.val-neg { font-weight: 700; color: #B91C1C; }
.val-neu { font-weight: 600; color: #334155; }
.sig-yes { color: #15803D; font-weight: 600; font-size: 13px; }

/* ── horizontal bar chart ── */
.hbar-chart { display: flex; flex-direction: column; gap: 14px; }
.hbar-row   { display: flex; align-items: center; gap: 12px; }
.hbar-label { width: 130px; font-size: 12px; font-weight: 500; color: #334155; text-align: right; flex-shrink: 0; }
.hbar-lines { flex: 1; display: flex; flex-direction: column; gap: 4px; }
.hbar-line  { display: flex; align-items: center; gap: 8px; }
.hbar-fill  { height: 9px; border-radius: 2px; min-width: 3px; transition: width 0.3s; }
.hbar-adopted { background: #1D4ED8; }
.hbar-churned { background: #CBD5E1; }
.hbar-pct   { font-size: 11px; color: #94A3B8; width: 34px; flex-shrink: 0; }
.hbar-pct.main { color: #1D4ED8; font-weight: 700; }

/* ── client table ── */
.data-table { width: 100%; border-collapse: collapse; }
.data-table thead th {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: #64748B;
  padding: 10px 14px;
  background: #F8FAFC;
  border-top: 1px solid #CBD5E1;
  border-bottom: 1px solid #CBD5E1;
  text-align: left;
}
.data-table tbody td {
  padding: 10px 14px;
  font-size: 13px;
  border-bottom: 1px solid #F1F5F9;
  vertical-align: middle;
}
.data-table tbody tr:hover td { background: #F8FAFC; }
.data-table tbody tr:last-child td { border-bottom: none; }
.client-id { font-weight: 600; color: #0F172A; font-size: 13px; }
.client-sub { font-size: 11px; color: #94A3B8; }

/* ── seg cards ── */
.seg-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 24px; }
.seg-card { border-radius: 10px; padding: 16px 18px; }
.seg-card.s1 { background: #FEF2F2; border: 1px solid #FECACA; }
.seg-card.s2 { background: #FFFBEB; border: 1px solid #FDE68A; }
.seg-card.s3 { background: #F0FDF4; border: 1px solid #BBF7D0; }
.seg-card.s4 { background: #EFF6FF; border: 1px solid #BFDBFE; }
.seg-n    { font-size: 26px; font-weight: 700; letter-spacing: -1px; line-height: 1; }
.s1 .seg-n { color: #B91C1C; } .s2 .seg-n { color: #B45309; }
.s3 .seg-n { color: #15803D; } .s4 .seg-n { color: #1D4ED8; }
.seg-title { font-size: 12px; font-weight: 600; margin-top: 6px; }
.s1 .seg-title { color: #991B1B; } .s2 .seg-title { color: #92400E; }
.s3 .seg-title { color: #14532D; } .s4 .seg-title { color: #1E40AF; }
.seg-desc  { font-size: 11px; color: #64748B; margin-top: 2px; }

/* ── rec card ── */
.rec-card {
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 12px;
}
.rec-card.r1 { background: #FEF2F2; border: 1px solid #FECACA; }
.rec-card.r2 { background: #FFFBEB; border: 1px solid #FDE68A; }
.rec-card.r3 { background: #F0FDF4; border: 1px solid #BBF7D0; }
.rec-header { font-size: 13px; font-weight: 700; margin-bottom: 10px; }
.r1 .rec-header { color: #991B1B; }
.r2 .rec-header { color: #92400E; }
.r3 .rec-header { color: #14532D; }
.rec-actions { display: flex; flex-direction: column; gap: 7px; }
.rec-action { display: flex; gap: 10px; font-size: 12px; color: #334155; line-height: 1.5; }
.rec-action-num {
  min-width: 20px; height: 20px; border-radius: 50%;
  background: rgba(0,0,0,0.08); display: flex; align-items: center;
  justify-content: center; font-size: 10px; font-weight: 700;
  flex-shrink: 0; margin-top: 1px;
}
.rec-evidence { font-size: 11px; color: #94A3B8; margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(0,0,0,0.06); }

/* ── streamlit tab styling ── */
button[data-baseweb="tab"] {
  font-size: 13px !important;
  font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  font-weight: 600 !important;
}

/* ── selectbox / filters — force light theme ── */
[data-testid="stSelectbox"] label {
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: #64748B !important;
}
div[data-baseweb="select"] > div:first-child {
  background-color: #FFFFFF !important;
  border-color: #CBD5E1 !important;
  border-radius: 8px !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
  color: #0F172A !important;
  background-color: transparent !important;
}
div[data-baseweb="select"] svg { color: #64748B !important; }
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"] {
  background-color: #FFFFFF !important;
  border: 1px solid #E2E8F0 !important;
  border-radius: 8px !important;
  box-shadow: 0 4px 16px rgba(0,0,0,0.08) !important;
}
li[role="option"] {
  background-color: #FFFFFF !important;
  color: #0F172A !important;
}
li[role="option"]:hover {
  background-color: #F1F5F9 !important;
}
li[role="option"][aria-selected="true"] {
  background-color: #EFF6FF !important;
  color: #1D4ED8 !important;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def badge(level) -> str:
    level = str(level) if pd.notna(level) else "—"
    cls = {"Crítico": "critico", "Alto": "alto", "Medio": "medio", "Bajo": "bajo"}.get(level, "bajo")
    return f'<span class="badge badge-{cls}">{level}</span>'


def score_bar(score: float) -> str:
    pct = int(score * 100)
    if score < 0.3:
        color = "#B91C1C"
    elif score < 0.5:
        color = "#D97706"
    elif score < 0.7:
        color = "#65A30D"
    else:
        color = "#1D4ED8"
    return (
        f'<div class="score-wrap">'
        f'<div class="score-track"><div class="score-fill" style="width:{pct}%;background:{color}"></div></div>'
        f'<div class="score-num">{pct}%</div>'
        f'</div>'
    )


def kpi_card(label: str, value, sub: str = "", color: str = "") -> str:
    cls = f"kpi-{color}" if color else ""
    return (
        f'<div class="kpi-card {cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{"<div class=\'kpi-sub\'>" + sub + "</div>" if sub else ""}'
        f'</div>'
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def insight(html: str) -> None:
    st.markdown(f'<div class="insight-box">{html}</div>', unsafe_allow_html=True)


FEATURE_LABELS = {
    "obs_sessions_count_w14": "Sesiones totales",
    "obs_unique_features_w14": "Funcionalidades distintas",
    "obs_active_days_w14": "Días activos",
    "obs_active_days_ratio_w14": "Ratio días activos",
    "obs_events_late_half_w14": "Eventos 2ª semana",
    "obs_job_close_out_count_w14": "Job Close Out",
    "obs_time_count_w14": "Uso Time",
    "fma_penetration_30d": "% colegas en FMA",
    "CORE_HEALTH_SCORE": "Health Score cliente",
}

FUNC_COLS = {
    "Job Close Out": "obs_job_close_out_count_w14",
    "Time":         "obs_time_count_w14",
    "Payment":      "obs_payment_count_w14",
    "Estimate":     "obs_estimate_count_w14",
    "Form":         "obs_form_count_w14",
    "Media":        "obs_media_count_w14",
}

ACTION_MAP = {
    "Crítico": "Contactar cliente",
    "Alto":    "Atención prioritaria",
    "Medio":   "Monitorear",
    "Bajo":    "En buen camino",
}


# ── data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    feat   = pd.read_parquet(DATA_DIR / "analytical_dataset.parquet")
    tenant = pd.read_parquet(DATA_DIR / "tenant_scores.parquet")
    shap   = pd.read_parquet(DATA_DIR / "shap_values.parquet")
    imp    = pd.read_parquet(DATA_DIR / "feature_importance.parquet")
    return feat, tenant, shap, imp


def last_updated() -> str:
    p = DATA_DIR / "last_updated.txt"
    return p.read_text().strip()[:16].replace("T", " ") if p.exists() else "—"


try:
    feat, tenant, shap_df, feat_imp = load_data()
    data_ready = True
except FileNotFoundError:
    data_ready = False


# ── navbar ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fma-navbar">
  <div class="fma-brand">FMA <em>Adoption</em> <span>/</span> Intelligence</div>
  <div class="fma-divider"></div>
  <div class="fma-sub">ServiceTitan · Migración TMA → FMA</div>
  <div class="fma-updated">Actualizado: {last_updated()}</div>
</div>
""", unsafe_allow_html=True)

if not data_ready:
    st.error("⚠️ Ejecuta `python pipeline.py` para generar los datos analíticos antes de abrir el tablero.")
    st.stop()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📋  Factores de Adopción",
    "🎯  Puntuación de Riesgo",
    "🗺️  Guía de Migración",
])


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 1 — Factores de Adopción
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # filtros
    section_label("Filtros")
    fc1, fc2, fc3 = st.columns(3)
    inds  = ["Todas"] + sorted(feat["TENANT_PRIMARY_INDUSTRY"].dropna().unique().tolist())
    sizes = ["Todos"] + sorted(feat["TENANT_SIZE"].dropna().unique().tolist())
    plats = ["Todas"] + sorted(feat["device_operating_system_mode"].dropna().unique().tolist())
    sel_ind  = fc1.selectbox("Industria", inds,  key="v1_ind")
    sel_size = fc2.selectbox("Tamaño del tenant", sizes, key="v1_size")
    sel_plat = fc3.selectbox("Plataforma del dispositivo", plats, key="v1_plat")

    df1 = feat.copy()
    if sel_ind  != "Todas": df1 = df1[df1["TENANT_PRIMARY_INDUSTRY"] == sel_ind]
    if sel_size != "Todos": df1 = df1[df1["TENANT_SIZE"] == sel_size]
    if sel_plat != "Todas": df1 = df1[df1["device_operating_system_mode"] == sel_plat]

    adopted  = df1[df1["label_retained"] == 1]
    abandoned = df1[df1["label_retained"] == 0]
    ret_pct  = len(adopted) / len(df1) * 100 if len(df1) > 0 else 0

    # KPI hero
    section_label("Indicadores globales")
    st.markdown(
        f'<div class="kpi-grid kpi-grid-4">'
        + kpi_card("Adoptantes sostenidos", f"{len(adopted):,}", "≥15 días activos en 60 días", "green")
        + kpi_card("Abandonaron FMA", f"{len(abandoned):,}", "< umbral de retención", "red")
        + kpi_card("Tasa de retención", f"{ret_pct:.0f}%", "en la muestra filtrada", "blue")
        + kpi_card("Técnicos en muestra", f"{len(df1):,}")
        + "</div>",
        unsafe_allow_html=True,
    )

    def _ratio(a_col, b_col):
        av = pd.to_numeric(a_col, errors="coerce").median()
        bv = pd.to_numeric(b_col, errors="coerce").median()
        return av / bv if bv and bv > 0 else None

    if len(adopted) >= 2 and len(abandoned) >= 2:
        ev_x   = _ratio(adopted["obs_events_late_half_w14"], abandoned["obs_events_late_half_w14"])
        time_x = _ratio(adopted["obs_time_count_w14"], abandoned["obs_time_count_w14"])
        adr_ad = pd.to_numeric(adopted["obs_active_days_ratio_w14"], errors="coerce").mean() * 100
        adr_ab = pd.to_numeric(abandoned["obs_active_days_ratio_w14"], errors="coerce").mean() * 100
        ev_str   = f"~{ev_x:.0f}×" if ev_x else "N/A"
        time_str = f"~{time_x:.0f}×" if time_x else "N/A"
        insight(
            f"Los técnicos que adoptan FMA generan <strong>{ev_str} más eventos en la segunda semana</strong> "
            f"y usan la funcionalidad Time <strong>{time_str} más</strong> que los que abandonan. "
            f"Ratio de días activos: {adr_ad:.1f}% (adoptantes) vs {adr_ab:.1f}% (abandono)."
        )
    else:
        insight("Selecciona un filtro con al menos 2 técnicos en cada grupo para ver el análisis comparativo.")

    # tabla comparativa
    section_label("Perfil comparativo — Primeras 2 semanas en FMA")
    NUM_CMP = [
        "obs_sessions_count_w14", "obs_unique_features_w14", "obs_active_days_w14",
        "obs_active_days_ratio_w14", "obs_events_late_half_w14",
        "obs_job_close_out_count_w14", "obs_time_count_w14",
        "fma_penetration_30d", "CORE_HEALTH_SCORE",
    ]
    CAT_CMP = ["app_version_max", "FMA_MIGRATION_STATUS_30D"]

    rows_html = ""
    n_sig = 0
    for c in NUM_CMP:
        if c not in df1.columns: continue
        av = pd.to_numeric(adopted[c],   errors="coerce").dropna().values
        bv = pd.to_numeric(abandoned[c], errors="coerce").dropna().values
        if len(av) < 2 or len(bv) < 2: continue
        _, p = ttest_ind(av, bv, equal_var=False)
        sig = p < 0.05
        if sig: n_sig += 1
        pstr = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
        av_fmt = f"{av.mean():.2f}"
        bv_fmt = f"{bv.mean():.2f}"
        diff_cls = "val-pos" if av.mean() > bv.mean() else "val-neg"
        sig_str = '<span class="sig-yes">✓</span>' if sig else '<span style="color:#CBD5E1">—</span>'
        rows_html += (
            f"<tr><td>{FEATURE_LABELS.get(c,c)}</td>"
            f'<td class="val-pos">{av_fmt}</td>'
            f'<td class="val-neg">{bv_fmt}</td>'
            f"<td>{pstr}</td>"
            f"<td style='text-align:center'>{sig_str}</td></tr>"
        )

    for c in CAT_CMP:
        if c not in df1.columns: continue
        try:
            ct = pd.crosstab(df1[c].fillna("Unknown"), df1["label_retained"])
            if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                chi2s, p, *_ = chi2_contingency(ct)
                sig = p < 0.05
                if sig: n_sig += 1
                pstr = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
                m_ad = df1.loc[df1["label_retained"]==1, c].mode()
                m_ab = df1.loc[df1["label_retained"]==0, c].mode()
                sig_str = '<span class="sig-yes">✓</span>' if sig else '<span style="color:#CBD5E1">—</span>'
                rows_html += (
                    f"<tr><td>{c}</td>"
                    f'<td class="val-pos">{m_ad.iloc[0] if len(m_ad) else "—"}</td>'
                    f'<td class="val-neg">{m_ab.iloc[0] if len(m_ab) else "—"}</td>'
                    f"<td>{pstr}</td>"
                    f"<td style='text-align:center'>{sig_str}</td></tr>"
                )
        except Exception:
            pass

    r2_ok = "✅" if n_sig >= 5 else "⚠️"
    st.markdown(
        f'<div class="fma-card">'
        f'<div class="card-title">Factores discriminantes &nbsp; '
        f'<span style="font-size:11px;color:#64748B;font-weight:400">'
        f'R2: {n_sig} factores p&lt;0.05 (umbral ≥5) {r2_ok}</span></div>'
        f'<table class="comp-table" style="margin-top:12px">'
        f'<thead><tr>'
        f'<th>Variable</th>'
        f'<th class="th-green">Adoptante</th>'
        f'<th class="th-red">Abandono</th>'
        f'<th>p-valor</th>'
        f'<th style="text-align:center">Sig.</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    # gráfico de funcionalidades
    section_label("% de uso de funcionalidades en semana 1")
    func_rows = []
    for fname, col in FUNC_COLS.items():
        if col not in df1.columns: continue
        p_ad = float((adopted[col] > 0).sum() / len(adopted) * 100) if len(adopted) else 0
        p_ab = float((abandoned[col] > 0).sum() / len(abandoned) * 100) if len(abandoned) else 0
        func_rows.append((fname, p_ad, p_ab))

    if func_rows:
        bars_html = ""
        for fname, p_ad, p_ab in func_rows:
            max_w = max(p_ad, p_ab, 1)
            w_ad = int(p_ad / max_w * 160)
            w_ab = int(p_ab / max_w * 160)
            bars_html += (
                f'<div class="hbar-row">'
                f'<div class="hbar-label">{fname}</div>'
                f'<div class="hbar-lines">'
                f'<div class="hbar-line"><div class="hbar-fill hbar-adopted" style="width:{w_ad}px"></div>'
                f'<div class="hbar-pct main">{p_ad:.0f}%</div></div>'
                f'<div class="hbar-line"><div class="hbar-fill hbar-churned" style="width:{w_ab}px"></div>'
                f'<div class="hbar-pct">{p_ab:.0f}%</div></div>'
                f'</div></div>'
            )
        st.markdown(
            f'<div class="fma-card">'
            f'<div style="display:flex;gap:18px;font-size:12px;color:#64748B;margin-bottom:16px">'
            f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#1D4ED8;margin-right:5px"></span>Adoptante</div>'
            f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:#CBD5E1;margin-right:5px"></span>Abandono</div>'
            f'</div>'
            f'<div class="hbar-chart">{bars_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # SHAP global
    section_label("Top-10 variables — Importancia SHAP global (modelo RF)")
    top10 = feat_imp.head(10).copy()
    top10["feature"] = top10["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
    top10_sorted = top10.sort_values("mean_abs_shap")
    max_shap = top10_sorted["mean_abs_shap"].max()
    shap_html = ""
    for _, row in top10_sorted.iterrows():
        w = int(row["mean_abs_shap"] / max_shap * 220)
        shap_html += (
            f'<div class="hbar-row">'
            f'<div class="hbar-label">{row["feature"]}</div>'
            f'<div class="hbar-lines"><div class="hbar-line">'
            f'<div class="hbar-fill" style="width:{w}px;background:#6D28D9"></div>'
            f'<div class="hbar-pct" style="color:#6D28D9;font-weight:600">{row["mean_abs_shap"]:.4f}</div>'
            f'</div></div></div>'
        )
    st.markdown(
        f'<div class="fma-card"><div class="hbar-chart">{shap_html}</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 2 — Puntuación de Riesgo
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    # filtros
    section_label("Filtros")
    f2c1, f2c2, f2c3 = st.columns(3)
    risk_opts = ["Todos", "Crítico", "Alto", "Medio", "Bajo"]
    mig_vals  = ["Todos"] + sorted(tenant["FMA_MIGRATION_STATUS_30D"].dropna().unique().tolist()) \
                if "FMA_MIGRATION_STATUS_30D" in tenant.columns else ["Todos"]
    ind2_vals = ["Todas"] + sorted(tenant["TENANT_PRIMARY_INDUSTRY"].dropna().unique().tolist()) \
                if "TENANT_PRIMARY_INDUSTRY" in tenant.columns else ["Todas"]

    sel_risk = f2c1.selectbox("Nivel de riesgo", risk_opts, key="v2_risk")
    sel_mig  = f2c2.selectbox("Estado de migración", mig_vals, key="v2_mig")
    sel_ind2 = f2c3.selectbox("Industria", ind2_vals, key="v2_ind")

    t2 = tenant.copy()
    if sel_risk != "Todos":
        t2 = t2[t2["tenant_risk_level"].astype(str) == sel_risk]
    if sel_mig != "Todos" and "FMA_MIGRATION_STATUS_30D" in t2.columns:
        t2 = t2[t2["FMA_MIGRATION_STATUS_30D"] == sel_mig]
    if sel_ind2 != "Todas" and "TENANT_PRIMARY_INDUSTRY" in t2.columns:
        t2 = t2[t2["TENANT_PRIMARY_INDUSTRY"] == sel_ind2]
    t2 = t2.sort_values("tenant_score_mean").reset_index(drop=True)

    n_crit = int((t2["tenant_risk_level"].astype(str) == "Crítico").sum())
    n_high = int((t2["tenant_risk_level"].astype(str) == "Alto").sum())

    stag_all = tenant[pd.to_numeric(tenant["retention_real"], errors="coerce") < 0.5]
    stag_risk_pct = (
        int(stag_all["tenant_risk_level"].isin(["Crítico", "Alto"]).sum() / len(stag_all) * 100)
        if len(stag_all) > 0 else 0
    )

    # KPI hero
    section_label("Indicadores")
    st.markdown(
        f'<div class="kpi-grid kpi-grid-5">'
        + kpi_card("Clientes riesgo crítico", str(n_crit), "score medio < 30%", "red")
        + kpi_card("Clientes riesgo alto", str(n_high), "score medio 30–50%", "amber")
        + kpi_card("Total clientes en muestra", str(len(t2)))
        + kpi_card("Estancados en riesgo alto/crítico", f"{stag_risk_pct}%", "retención real < 50% · R1", "red")
        + kpi_card("AUC-ROC del modelo", f"{AUC_VAL:.3f}", "R5: umbral ≥0.70 ✅", "green")
        + "</div>",
        unsafe_allow_html=True,
    )

    # tabla de clientes
    section_label("Clientes ordenados por nivel de riesgo")

    rows_t = ""
    for _, row in t2.iterrows():
        fma_pct = f"{row['fma_penetration_30d']*100:.0f}%" if pd.notna(row.get("fma_penetration_30d")) else "—"
        ind_txt = str(row.get("TENANT_PRIMARY_INDUSTRY", "—"))[:20] if pd.notna(row.get("TENANT_PRIMARY_INDUSTRY")) else "—"
        action  = ACTION_MAP.get(str(row.get("tenant_risk_level", "")), "—")
        rows_t += (
            f"<tr>"
            f'<td><span class="client-id">#{row["TENANT_ID"]}</span>'
            f'<br><span class="client-sub">{ind_txt}</span></td>'
            f"<td>{badge(row.get('tenant_risk_level'))}</td>"
            f"<td>{score_bar(row['tenant_score_mean'])}</td>"
            f"<td>{fma_pct}</td>"
            f"<td>{int(row['n_techs'])}</td>"
            f"<td>{int(row.get('n_techs_critical',0))}</td>"
            f'<td><span style="font-size:12px;color:#64748B">{action}</span></td>'
            f"</tr>"
        )

    st.markdown(
        f'<div class="fma-card" style="padding:0;overflow:hidden">'
        f'<table class="data-table">'
        f'<thead><tr>'
        f'<th>Cliente</th><th>Riesgo</th><th>Score retención</th>'
        f'<th>% en FMA</th><th>Técnicos</th><th>En riesgo crítico</th><th>Acción sugerida</th>'
        f'</tr></thead>'
        f'<tbody>{rows_t}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    # detalle técnico
    section_label("Detalle de técnicos por cliente")
    if len(t2) == 0:
        st.info("Sin clientes que coincidan con los filtros.")
    else:
        sel_tenant = st.selectbox(
            "Seleccionar cliente",
            t2["TENANT_ID"].tolist(),
            format_func=lambda x: f"Cliente #{x}",
            key="v2_sel",
        )
        tf = feat[feat["TENANT_ID"] == sel_tenant].copy()
        ts = shap_df[shap_df["TECHNICIAN_ID"].isin(tf["TECHNICIAN_ID"])].copy()
        shap_cols = ["TECHNICIAN_ID", "factor_1", "shap_1", "factor_2", "shap_2", "factor_3", "shap_3"]
        tm = tf[["TECHNICIAN_ID", "pred_score", "risk_level", "label_retained"]].merge(
            ts[[c for c in shap_cols if c in ts.columns]], on="TECHNICIAN_ID", how="left"
        ).sort_values("pred_score").reset_index(drop=True)

        def clean_f(val):
            return str(val).replace("num__", "").replace("cat__", "") if pd.notna(val) else None

        tech_rows = ""
        for _, r in tm.iterrows():
            f1 = clean_f(r.get("factor_1")) or "—"
            f2 = clean_f(r.get("factor_2"))
            f3 = clean_f(r.get("factor_3"))
            factors_html = f'<div style="font-size:12px;font-weight:600;color:#1E293B">{f1}</div>'
            if f2:
                factors_html += f'<div style="font-size:11px;color:#64748B;margin-top:2px">{f2}</div>'
            if f3:
                factors_html += f'<div style="font-size:11px;color:#94A3B8;margin-top:1px">{f3}</div>'
            real_lbl = "✓ Retenido" if r["label_retained"] == 1 else "✗ Abandonó"
            lbl_color = "#15803D" if r["label_retained"] == 1 else "#B91C1C"
            tech_rows += (
                f"<tr>"
                f'<td><span style="font-family:monospace;font-size:12px">{r["TECHNICIAN_ID"]}</span></td>'
                f"<td>{score_bar(r['pred_score'])}</td>"
                f"<td>{badge(r['risk_level'])}</td>"
                f'<td style="line-height:1.5">{factors_html}</td>'
                f'<td><span style="font-size:11px;font-weight:600;color:{lbl_color}">{real_lbl}</span></td>'
                f'<td><span style="font-size:12px;color:#64748B">{ACTION_MAP.get(str(r.get("risk_level","")),"—")}</span></td>'
                f"</tr>"
            )

        st.markdown(
            f'<div class="fma-card" style="padding:0;overflow:hidden">'
            f'<table class="data-table">'
            f'<thead><tr>'
            f'<th>Técnico ID</th><th>Prob. retención</th><th>Nivel riesgo</th>'
            f'<th>Top-3 factores SHAP</th><th>Resultado real</th><th>Acción</th>'
            f'</tr></thead>'
            f'<tbody>{tech_rows}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# VISTA 3 — Guía de Migración
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    # banner condicional
    if AUC_VAL >= 0.70:
        st.markdown(
            f'<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;'
            f'padding:12px 18px;font-size:13px;color:#14532D;margin-bottom:16px">'
            f'✅ <strong>Recomendaciones activadas</strong> · AUC = {AUC_VAL:.3f} ≥ 0.70 · '
            f'Las recomendaciones están respaldadas por el modelo predictivo.</div>',
            unsafe_allow_html=True,
        )

    # segmentación
    seg_data = tenant.copy()
    seg_data["pct_fma"] = seg_data.get("fma_penetration_30d", pd.Series(dtype=float))

    def classify_seg(row):
        pct = row.get("pct_fma", np.nan)
        if pd.isna(pct): return "Sin dato"
        if pct < 0.20:   return "Migrar con prioridad"
        if pct < 0.40:   return "Activar funcionalidades clave"
        if pct < 0.70:   return "Monitorear adopción"
        return "En buen camino"

    seg_data["segmento"] = seg_data.apply(classify_seg, axis=1)

    # filtros Vista 3
    section_label("Filtros")
    v3c1, v3c2 = st.columns(2)
    seg_opts  = ["Todos", "Migrar con prioridad", "Activar funcionalidades clave",
                 "Monitorear adopción", "En buen camino"]
    ind3_vals = ["Todas"] + sorted(seg_data["TENANT_PRIMARY_INDUSTRY"].dropna().unique().tolist()) \
                if "TENANT_PRIMARY_INDUSTRY" in seg_data.columns else ["Todas"]
    sel_seg3 = v3c1.selectbox("Segmento", seg_opts, key="v3_seg")
    sel_ind3 = v3c2.selectbox("Industria", ind3_vals, key="v3_ind")

    if sel_seg3 != "Todos":
        seg_data = seg_data[seg_data["segmento"] == sel_seg3]
    if sel_ind3 != "Todas" and "TENANT_PRIMARY_INDUSTRY" in seg_data.columns:
        seg_data = seg_data[seg_data["TENANT_PRIMARY_INDUSTRY"] == sel_ind3]

    sc = seg_data["segmento"].value_counts()

    # KPI hero
    section_label("Indicadores")
    st.markdown(
        f'<div class="kpi-grid kpi-grid-2">'
        + kpi_card("Clientes en estado intermedio de migración", f"{len(tenant):,}", "con al menos 1 técnico nuevo en FMA")
        + kpi_card("Umbral de inflexión social", "40%", "≥40% técnicos en FMA = punto de no retorno", "purple")
        + "</div>",
        unsafe_allow_html=True,
    )

    # 4 segmentos
    section_label("Segmentos de acción")
    st.markdown(
        f'<div class="seg-grid">'
        f'<div class="seg-card s1"><div class="seg-n">{sc.get("Migrar con prioridad",0)}</div>'
        f'<div class="seg-title">Migrar con prioridad</div><div class="seg-desc">&lt; 20% en FMA</div></div>'
        f'<div class="seg-card s2"><div class="seg-n">{sc.get("Activar funcionalidades clave",0)}</div>'
        f'<div class="seg-title">Activar funcionalidades clave</div><div class="seg-desc">20 – 39% en FMA</div></div>'
        f'<div class="seg-card s3"><div class="seg-n">{sc.get("Monitorear adopción",0)}</div>'
        f'<div class="seg-title">Monitorear adopción</div><div class="seg-desc">40 – 69% en FMA</div></div>'
        f'<div class="seg-card s4"><div class="seg-n">{sc.get("En buen camino",0)}</div>'
        f'<div class="seg-title">En buen camino</div><div class="seg-desc">≥ 70% en FMA</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # tabla de priorización
    section_label("Tabla de priorización por cliente")
    prio = seg_data.sort_values("tenant_score_mean").reset_index(drop=True)
    prio_rows = ""
    for _, r in prio.iterrows():
        ind_t  = str(r.get("TENANT_PRIMARY_INDUSTRY","—"))[:18] if pd.notna(r.get("TENANT_PRIMARY_INDUSTRY")) else "—"
        fma_p  = f"{r['pct_fma']*100:.0f}%" if pd.notna(r.get("pct_fma")) else "—"
        ret_r  = f"{r['retention_real']*100:.0f}%" if pd.notna(r.get("retention_real")) else "—"
        seg    = str(r.get("segmento","—"))
        seg_color = {"Migrar con prioridad":"#B91C1C","Activar funcionalidades clave":"#B45309",
                     "Monitorear adopción":"#15803D","En buen camino":"#1D4ED8"}.get(seg,"#64748B")
        prio_rows += (
            f"<tr>"
            f'<td><span class="client-id">#{r["TENANT_ID"]}</span><br><span class="client-sub">{ind_t}</span></td>'
            f'<td style="font-size:12px;font-weight:600;color:{seg_color}">{seg}</td>'
            f"<td>{fma_p}</td>"
            f"<td>{score_bar(r['tenant_score_mean'])}</td>"
            f"<td>{ret_r}</td>"
            f"</tr>"
        )

    st.markdown(
        f'<div class="fma-card" style="padding:0;overflow:hidden">'
        f'<table class="data-table"><thead><tr>'
        f'<th>Cliente</th><th>Segmento</th><th>% en FMA</th>'
        f'<th>Score agregado</th><th>Retención real</th>'
        f'</tr></thead><tbody>{prio_rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )

    # recomendaciones
    section_label("Recomendaciones por segmento")
    st.markdown("""
<div class="rec-card r1">
  <div class="rec-header">🔴 Segmento 1 — Migrar con prioridad (&lt; 20% en FMA)</div>
  <div class="rec-actions">
    <div class="rec-action"><div class="rec-action-num">1</div>
      <div><strong>Priorizar flujos de mayor retención:</strong> activar Job Close Out y Time en las primeras sesiones.
      Estos son los dos factores SHAP dominantes de retención en este segmento.</div></div>
    <div class="rec-action"><div class="rec-action-num">2</div>
      <div><strong>Coordinar con Customer Success:</strong> asignar un CSM dedicado para los próximos 30 días.</div></div>
    <div class="rec-action"><div class="rec-action-num">3</div>
      <div><strong>Evaluar viabilidad:</strong> si el cliente tiene &lt;5 técnicos activos en FMA,
      evaluar si la migración es viable sin un plan de incentivos explícito.</div></div>
  </div>
  <div class="rec-evidence">Respaldo: SHAP #1 = obs_job_close_out_count_w14 · Q1 retención = 0%</div>
</div>
<div class="rec-card r2">
  <div class="rec-header">🟠 Segmento 2 — Activar funcionalidades clave (20 – 39% en FMA)</div>
  <div class="rec-actions">
    <div class="rec-action"><div class="rec-action-num">1</div>
      <div><strong>Activar funcionalidades según factor SHAP dominante:</strong> consultar Vista 2
      para el factor principal de cada técnico en riesgo.</div></div>
    <div class="rec-action"><div class="rec-action-num">2</div>
      <div><strong>Acelerar incorporación:</strong> superar el umbral del 40% en las próximas 4 semanas
      para activar el efecto de red.</div></div>
    <div class="rec-action"><div class="rec-action-num">3</div>
      <div><strong>Seguimiento bisemanal:</strong> monitorear obs_active_days_ratio_w14 como indicador adelantado.</div></div>
  </div>
  <div class="rec-evidence">Respaldo: Q2 retención = 34.6% · umbral inflexión = 40%</div>
</div>
<div class="rec-card r3">
  <div class="rec-header">🟢 Segmento 3 — Monitorear y consolidar (≥ 40% en FMA)</div>
  <div class="rec-actions">
    <div class="rec-action"><div class="rec-action-num">1</div>
      <div><strong>Mantener ritmo:</strong> no interrumpir el ciclo de adopción. Evitar cambios de versión
      de app en los primeros 60 días del técnico nuevo.</div></div>
    <div class="rec-action"><div class="rec-action-num">2</div>
      <div><strong>Recuperar técnicos en zona de riesgo:</strong> seguimiento bisemanal de técnicos con
      score 0.30–0.50 (riesgo alto/medio).</div></div>
    <div class="rec-action"><div class="rec-action-num">3</div>
      <div><strong>Documentar como caso de éxito:</strong> usar estos tenants como referencia para segmentos 1 y 2.</div></div>
  </div>
  <div class="rec-evidence">Respaldo: Q3 retención = 53.8% · Q4 retención = 78.6%</div>
</div>
""", unsafe_allow_html=True)
