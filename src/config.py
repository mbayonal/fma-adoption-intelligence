"""Constantes del pipeline para entrega_3.

Todas las rutas son relativas al directorio raíz de entrega_3/.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR
MOD_DIR = ROOT / "models"

RAW_TENANTS = RAW_DIR / "fma_tenant_data_sample.csv"
RAW_TECHS = RAW_DIR / "fma_technician_data_sample.csv"
RAW_EVENTS = RAW_DIR / "fma_fs_event_data_sample.csv"

RF_MODEL = MOD_DIR / "rf_model.joblib"
RF_METADATA = MOD_DIR / "rf_model_metadata.json"

RANDOM_STATE = 42
OBSERVATION_WINDOW_DAYS = 14
LABEL_GAP_DAYS = 0
LABEL_WINDOW_DAYS = 60
LABEL_ACTIVE_DAYS_THRESHOLD = 15
MIN_OBSERVABLE_DAYS_AFTER_OBS = 14

VALID_TENANT_STATUSES = ("Success", "Live", "Onboarding")
VALID_DEVICE_SOURCES = ("FMA", "TMA")

TRAIN_PERCENTILE_CUTOFF = 0.60
N_FEATURES_SELECTED = 20
SELECTION_RF_WEIGHT = 0.6
SELECTION_FILTER_WEIGHT = 0.4
VIF_THRESHOLD = 10.0
CV_N_SPLITS = 3
TUNING_N_ITER = 50
TUNING_SCORING = "roc_auc"
