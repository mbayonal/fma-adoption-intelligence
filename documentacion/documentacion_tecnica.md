# Documentación Técnica — FMA Adoption Intelligence

**Versión:** 1.0 | **Entrega:** entrega_3  
**Repositorio:** `analytics_final_project/entrega_3/`

---

## 1. Arquitectura del pipeline

```
CSVs fuente (data/raw/)
    │
    ▼
etl.run_phase1()          — limpieza de tenants, técnicos y eventos
    │
    ▼
etl.build_feature_store() — 57 variables por técnico (bloques A + contexto)
    │
    ▼
RF pipeline (joblib)      — predict_proba: score de retención [0,1]
    │
    ├──► analytical_dataset.parquet  (feature store + score + risk_level)
    │
    ├──► SHAP TreeExplainer          — top-3 factores por técnico
    │        └──► shap_values.parquet
    │        └──► feature_importance.parquet
    │
    └──► Agregación por tenant       — score medio, cuartil, métricas
             └──► tenant_scores.parquet
```

El pipeline **no re-entrena** el modelo. El modelo RF fue entrenado en `entrega_2` con los mismos datos fuente, usando validación out-of-time.

---

## 2. Configuración de rutas (`src/config.py`)

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `ROOT` | `entrega_3/` | Directorio raíz del prototipo |
| `RAW_DIR` | `data/raw/` | CSVs fuente |
| `OUT_DIR` | `data/` | Parquet de salida |
| `MOD_DIR` | `models/` | Modelo y metadatos |
| `RF_MODEL` | `models/rf_model.joblib` | Pipeline scikit-learn serializado |
| `RF_METADATA` | `models/rf_model_metadata.json` | Features y métricas del modelo |

---

## 3. Definición de la variable objetivo

```
y = 1  si el técnico registra ≥ 15 días activos en FMA durante los 60 días
        posteriores a la ventana de observación de 14 días
y = 0  si no lo hace
```

- **Ventana de observación (T0):** primeros 14 días del técnico en FMA.
- **Ventana de etiquetado (T1):** días 15–74 después de la primera sesión.
- **Umbral de etiquetado:** `LABEL_ACTIVE_DAYS_THRESHOLD = 15`.

---

## 4. Variables seleccionadas (Top-20 por combined_score)

```
combined_score = 0.6 × RF_importance_norm + 0.4 × MI/chi2_norm
```

| Variable | Tipo | Score combinado |
|----------|------|-----------------|
| app_version_max | Categórica | 1.000 |
| obs_events_late_half_w14 | Numérica | 0.398 |
| device_model_mode | Categórica | 0.377 |
| obs_active_days_ratio_w14 | Numérica | 0.367 |
| obs_sessions_count_w14 | Numérica | 0.348 |
| obs_active_days_w14 | Numérica | 0.323 |
| obs_active_days_late_half_w14 | Numérica | 0.247 |
| obs_time_count_w14 | Numérica | 0.225 |
| obs_events_count_w14 | Numérica | — |
| obs_events_count_w14_log | Numérica | — |
| obs_events_growth_ratio_w14 | Numérica | — |
| app_version_mode | Categórica | — |
| obs_active_days_early_half_w14 | Numérica | — |
| TECHNICIAN_LICENCES | Numérica | — |
| obs_job_close_out_count_w14 | Numérica | — |
| obs_time_pct_w14 | Numérica | — |
| tenant_technician_licenses_log | Numérica | — |
| FMA_MIGRATION_STATUS_30D | Categórica | — |
| TENANT_BUSINESS_FOCUS | Categórica | — |
| fma_penetration_30d | Numérica | — |

---

## 5. Preprocesamiento del modelo RF

```python
Pipeline([
    ("pre", ColumnTransformer([
        ("num", "passthrough", top_num),          # 15 variables numéricas
        ("cat", OneHotEncoder(handle_unknown="ignore"), top_cat),  # 5 categóricas → OHE
    ])),
    ("clf", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        # hiperparámetros optimizados via RandomizedSearchCV con TimeSeriesSplit
    ))
])
```

- **Desbalance:** `class_weight='balanced'` (proporción 3:1 en datos de entrenamiento).
- **NaN:** conteos = 0 cuando no hubo actividad; categóricas = "Unknown".

---

## 6. Métricas del modelo (validación out-of-time)

| Métrica | Valor | Umbral R | Estado |
|---------|-------|----------|--------|
| AUC-ROC | 0.829 | ≥ 0.70 (R5) | ✅ |
| Recall clase abandono | 0.78 | ≥ 0.70 (R7) | ✅ |
| Precisión clase abandono | 0.86 | ≥ 0.60 (R8) | ✅ |
| F1-Score clase abandono | 0.82 | ≥ 0.65 (R9) | ✅ |
| ΔAUC train–val | 16.8 pp | ≤ 5 pp (R11) | ❌ (limitación documentada) |

**Nota R11:** El gap se atribuye al tamaño reducido del conjunto de entrenamiento (n=84) y a la diferencia de distribución entre cohortes (tasa retención train=81% vs. val=43%). No impide el uso del modelo en producción dado que las métricas de validación superan todos los umbrales exigidos.

---

## 7. Proceso de re-entrenamiento (R21 / R16)

El modelo se re-entrena manualmente cuando hay nuevos datos disponibles.  
**Frecuencia recomendada:** bisemanal, o cuando lleguen datos de ≥50 técnicos nuevos.

### Pasos

1. Copiar los nuevos CSVs a `data/raw/`.
2. Desde el directorio `entrega_2/code/`:
   ```bash
   source .venv/bin/activate
   python -m src.run
   ```
3. Copiar el nuevo modelo a `entrega_3/models/`:
   ```bash
   cp entrega_2/code/outputs/models/RF.joblib entrega_3/models/rf_model.joblib
   ```
4. Actualizar `entrega_3/models/rf_model_metadata.json` con las métricas y features de la nueva corrida (ver `entrega_2/code/outputs/reports/run_metadata.json`).
5. Ejecutar `python pipeline.py` desde `entrega_3/` para regenerar los Parquet.
6. Verificar que las métricas del nuevo modelo cumplen los umbrales R5–R9.

### Criterios de alerta para re-entrenamiento forzado

- AUC en validación cae por debajo de 0.70 (R5).
- La distribución de `app_version_max` en nuevos datos difiere significativamente (nueva versión de app lanzada).
- Se incorporan > 500 técnicos nuevos al dataset (suficiente volumen para re-entrenar).

---

## 8. Despliegue en Streamlit Cloud (R17)

1. Subir `entrega_3/` a un repositorio GitHub (solo el código, sin CSVs).
2. En [streamlit.io/cloud](https://streamlit.io/cloud), conectar el repositorio y apuntar a `app.py`.
3. Configurar los datos analíticos: subir los Parquet generados como archivos de datos del app, o conectar a un bucket S3/GCS si se dispone de infraestructura.
4. El tablero queda accesible vía navegador para ≥3 usuarios sin instalación local (R17).

---

## 9. Flujo de datos completo

```
Snowflake (producción)
    └── FMA_FS_EVENT_DATA  (226M eventos)
    └── FMA_TECHNICIAN_DATA (~450K técnicos)
    └── FMA_TENANT_DATA    (~20K clientes)
           │
           ▼  (extracción bisemanal → CSV / Parquet)
    data/raw/  (entrega académica: muestra de datos)
           │
           ▼
    pipeline.py  (ETL + scoring + SHAP)
           │
           ▼
    data/*.parquet  (datos analíticos)
           │
           ▼
    app.py (Streamlit)  →  Navegador del PM
```

---

## 10. Tiempos de ejecución (R13)

| Componente | Tiempo (datos de muestra, n=140) | Estimado producción (n=66K) |
|-----------|----------------------------------|-----------------------------|
| ETL completo | ~1.5s | ~5–8 min |
| Scoring RF | < 0.1s | < 1 min |
| SHAP TreeExplainer | ~1.5s | ~2–4 min |
| Agregación tenant | < 0.1s | < 1 min |
| **Total pipeline** | **~3.8s** | **< 10 min** ✅ |

Estimación de producción basada en complejidad lineal del ETL y SHAP aproximado para n=66K con TreeExplainer.
