# Manual de Usuario — FMA Adoption Intelligence

**Versión:** 1.0 | **Tablero:** Streamlit  
**Usuarios destinatarios:** Product Managers del equipo FMA en ServiceTitan

---

## 1. Instalación y puesta en marcha (ejecución local)

> Sigue estos pasos la primera vez que configures el entorno. Solo se requiere Python 3.10 o superior.

### Paso 1 — Crear el entorno virtual

```bash
# Desde la carpeta raíz del proyecto (entrega_3/)
python3 -m venv .venv
```

### Paso 2 — Instalar dependencias

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Instalar paquetes
pip install -r requirements.txt
```

### Paso 3 — Generar los datos analíticos

Ejecuta el pipeline una vez para producir los archivos Parquet que el tablero necesita:

```bash
python pipeline.py
```

Duración aproximada: **< 5 segundos** con los datos de muestra incluidos.  
Los archivos se guardan en `data/` automáticamente.

### Paso 4 — Lanzar el tablero

```bash
streamlit run app.py
```

El tablero se abre en `http://localhost:8501`.  
Puedes compartirlo en tu red local con la URL `Network URL` que imprime Streamlit.

### Actualización de datos

Para refrescar el análisis con nuevos datos:
1. Reemplaza los CSVs en `data/raw/`.
2. Ejecuta `python pipeline.py` nuevamente.
3. El tablero recargará los Parquet automáticamente.

---

## 2. Acceso al tablero (versión en la nube)

Si el equipo publicó el tablero en Streamlit Community Cloud, simplemente abre  
la URL proporcionada en el navegador. No se requiere instalación local ni conocimiento técnico.

---

## 3. Estructura general

El tablero tiene tres vistas accesibles mediante pestañas en la parte superior:

| Pestaña | Tipo | Pregunta que responde |
|---------|------|-----------------------|
| 📋 Factores de Adopción | Descriptivo | ¿Qué características distinguen al técnico que adopta FMA? |
| 🎯 Puntuación de Riesgo | Predictivo | ¿Qué clientes tienen mayor riesgo de migración fallida? |
| 🗺️ Guía de Migración | Prescriptivo | ¿Qué acciones tomar por segmento de cliente? |

La barra superior muestra la **fecha de última actualización** de los datos.

---

## 4. Vista 1 — Factores de Adopción

### 3.1 Filtros disponibles

| Filtro | Efecto |
|--------|--------|
| Industria | Restringe el análisis a clientes de una industria específica |
| Tamaño del tenant | Filtra por tamaño del cliente (Small, Medium, Large, Enterprise) |
| Plataforma del dispositivo | Filtra por sistema operativo del dispositivo del técnico |

Los filtros actualizan automáticamente todos los indicadores y gráficos de la vista.

### 3.2 Indicadores hero

- **Adoptantes sostenidos:** técnicos que completaron ≥15 días activos en los 60 días posteriores al período de observación.
- **Abandonaron FMA:** técnicos que no alcanzaron ese umbral.
- **Tasa de retención:** proporción de adoptantes sostenidos sobre el total.
- **Técnicos en muestra:** total de técnicos con datos suficientes para análisis.

### 3.3 Tabla comparativa

Muestra las diferencias entre los perfiles de técnicos que adoptaron vs. abandonaron FMA en sus primeras 2 semanas:
- **p-valor < 0.05** → diferencia estadísticamente significativa (✅)
- Las variables numéricas usan prueba t de Welch; las categóricas usan chi cuadrado.

### 3.4 Gráfico de funcionalidades

Muestra el porcentaje de técnicos que usó cada funcionalidad en la semana 1, separado por grupo (adoptante / abandono). Funcionalidades con mayor diferencia indican qué features impulsan la retención.

### 3.5 Top-10 variables SHAP

Importancia global del modelo Random Forest según el valor absoluto medio de SHAP. Las barras más largas indican las variables que más impactan las predicciones de riesgo.

---

## 5. Vista 2 — Puntuación de Riesgo

### 4.1 Filtros disponibles

| Filtro | Efecto |
|--------|--------|
| Nivel de riesgo del tenant | Filtra clientes por nivel de riesgo agregado |
| Estado de migración | Filtra por estado actual de migración (Partial, Minimal, Majority, Complete) |
| Industria | Filtra por industria del cliente |

### 4.2 Indicadores hero

- **Clientes riesgo crítico:** clientes donde el score agregado de retención es < 30%.
- **Clientes riesgo alto:** clientes con score entre 30% y 50%.
- **Total clientes:** total de clientes con técnicos nuevos en la muestra.
- **AUC-ROC del modelo:** capacidad discriminativa del modelo (0.829, umbral ≥ 0.70).

### 4.3 Tabla de clientes

Los clientes están ordenados de menor a mayor score de retención (los primeros requieren intervención más urgente). La columna **Score** tiene gradiente de color: rojo = riesgo alto, verde = bajo riesgo.

**Columnas:**
- **Score:** probabilidad media de retención de los técnicos del cliente (0 = máximo riesgo).
- **Riesgo:** clasificación del nivel de riesgo del tenant.
- **Estado migración:** porcentaje de técnicos ya en FMA.
- **En riesgo crítico / alto:** número de técnicos individuales en cada nivel.

### 4.4 Detalle de técnicos por cliente

1. Seleccionar un cliente del menú desplegable "Seleccionar cliente".
2. La tabla inferior muestra todos los técnicos del cliente ordenados por score.
3. **Factor principal (SHAP):** la variable que más contribuye al score de ese técnico específico. Úsala para articular la intervención concreta con Customer Success.

**Interpretación del score:**
| Score | Nivel | Acción sugerida |
|-------|-------|-----------------|
| < 30% | Crítico | Contactar cliente inmediatamente |
| 30–50% | Alto | Atención prioritaria en próximos 14 días |
| 50–70% | Medio | Monitorear en ciclo bisemanal |
| > 70% | Bajo | En buen camino — seguimiento estándar |

---

## 6. Vista 3 — Guía de Migración

### 5.1 Segmentos de cliente

Los clientes se clasifican según el porcentaje actual de técnicos que usan FMA:

| Segmento | Criterio | Prioridad |
|----------|----------|-----------|
| Migrar con prioridad | < 20% en FMA | Máxima |
| Activar funcionalidades clave | 20–39% en FMA | Alta |
| Monitorear adopción | 40–69% en FMA | Media |
| En buen camino | ≥ 70% en FMA | Estándar |

El **umbral de inflexión social (40%)** es el punto a partir del cual la adopción tiende a auto-sostenerse por efecto de red entre colegas.

### 5.2 Tabla de priorización

Clientes ordenados por score de riesgo dentro de cada segmento. Las columnas de porcentaje y retención real permiten validar si el modelo está prediciendo correctamente.

### 5.3 Tarjetas de recomendación

Cada tarjeta incluye el **respaldo evidencial** de la recomendación: métricas del modelo, factores SHAP dominantes y tasas de retención por cuartil observadas en los datos.

---

## 7. Preguntas frecuentes

**¿Los datos se actualizan solos?**  
No. Los datos se actualizan ejecutando `python pipeline.py` cada 14 días (bisemanal) o bajo demanda. La fecha de última actualización aparece en el encabezado del tablero.

**¿Los nombres de los clientes están visibles?**  
No. El tablero usa únicamente identificadores internos (IDs) y métricas agregadas. No se muestran nombres de clientes ni de técnicos (R20).

**¿Por qué un técnico tiene score bajo si está usando FMA?**  
El score predice retención a 60 días basándose en el comportamiento de las primeras 2 semanas. Un técnico que usa pocas funcionalidades o con baja frecuencia puede tener score bajo incluso si ha iniciado sesión.

**¿Qué hacer si el modelo tiene AUC < 0.70?**  
El tablero mostrará una advertencia en la Vista 3. En ese caso, las recomendaciones prescriptivas deben tomarse como orientativas hasta que el modelo sea re-entrenado con más datos.
