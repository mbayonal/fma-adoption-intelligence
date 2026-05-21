# FMA Adoption Intelligence — Prototipo Funcional (entrega_3)

Tablero analítico interactivo para anticipar qué técnicos de campo abandonarán FMA
y priorizar intervenciones por tenant.

**Proyecto:** Service Titan — Inteligencia de Adopción de FMA  
**Autores:** Andrés Alfonso, Daniela Uscátegui, Daniel Ricardo Marín, Manuel Alejandro Bayona  
**Curso:** MIAD PAAD | Universidad de los Andes

---

## Estructura

```
entrega_3/
├── app.py                       # Tablero Streamlit (3 vistas)
├── pipeline.py                  # ETL + scoring + SHAP + agregación tenant
├── requirements.txt
├── README.md
├── src/
│   ├── config.py                # Rutas y constantes
│   └── etl.py                   # ETL: limpieza y feature store
├── models/
│   ├── rf_model.joblib          # Modelo RF pre-entrenado (entrega_2)
│   └── rf_model_metadata.json  # Features seleccionadas + métricas
├── data/
│   ├── raw/                     # CSVs fuente (no se incluyen en zip)
│   │   ├── fma_fs_event_data_sample.csv
│   │   ├── fma_technician_data_sample.csv
│   │   └── fma_tenant_data_sample.csv
│   ├── analytical_dataset.parquet   # generado por pipeline.py
│   ├── tenant_scores.parquet        # generado por pipeline.py
│   ├── shap_values.parquet          # generado por pipeline.py
│   ├── feature_importance.parquet   # generado por pipeline.py
│   └── last_updated.txt             # generado por pipeline.py
└── documentacion/
    ├── manual_de_usuario.md
    └── documentacion_tecnica.md
```

---

## Instalación

```bash
cd entrega_3
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Uso

### 1. Ejecutar el pipeline (genera los datos analíticos)

```bash
python pipeline.py
```

Tiempo estimado: < 2 minutos con los datos de muestra.  
Salida: `data/*.parquet` + `data/last_updated.txt`

### 2. Lanzar el tablero

```bash
streamlit run app.py
```

El tablero abre en `http://localhost:8501`.

---

## Vistas del tablero

| Vista | Tipo | Requerimientos |
|-------|------|----------------|
| Factores de Adopción | Descriptivo | R2, R14.1 |
| Puntuación de Riesgo | Predictivo | R1, R3, R5–R10, R14.2 |
| Guía de Migración | Prescriptivo | R4, R14.3 |

---

## Métricas del modelo (Random Forest)

| Métrica | Valor | Umbral | Estado |
|---------|-------|--------|--------|
| AUC-ROC (validación) | 0.829 | ≥ 0.70 (R5) | ✅ |
| Recall clase abandono | 0.78 | ≥ 0.70 (R7) | ✅ |
| Precisión clase abandono | 0.86 | ≥ 0.60 (R8) | ✅ |
| F1-Score | 0.82 | ≥ 0.65 (R9) | ✅ |

---

## Paquetización para compartir

```bash
cd ..
zip -r entrega_3.zip entrega_3/ \
    --exclude "entrega_3/.venv/*" \
    --exclude "entrega_3/__pycache__/*" \
    --exclude "entrega_3/data/raw/*" \
    --exclude "entrega_3/data/*.parquet" \
    --exclude "entrega_3/data/last_updated.txt"
```

Los datos de muestra se comparten por separado por restricciones de confidencialidad.
