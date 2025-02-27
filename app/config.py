"""
Konfigurační soubor pro KNN-Math
"""

import os

# Cesty k adresářům
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "models", "knn_model.pkl")

# Parametry KNN klasifikátoru
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"  # uniform nebo distance

# Parametry předzpracování
IMAGE_SIZE = (32, 32)  # Výška a šířka po normalizaci
