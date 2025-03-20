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

# Konfigurace pro KNN model
KNN_K = 3
KNN_DISTANCE_METRIC = "euclidean"

# Konfigurace pro segmentaci
SEGMENTATION_THRESHOLD = 128
SEGMENTATION_MIN_AREA = 50

# Konfigurace pro neuronovou síť
NN_CONFIG = {
    # Model architecture
    "encoder_dim": 256,
    "decoder_dim": 512,
    "embedding_dim": 256,
    "attention_dim": 256,
    "dropout": 0.5,
    # Training
    "learning_rate": 3e-4,
    "batch_size": 32,
    "epochs": 30,
    "clip_gradient": 5.0,
    # Inference
    "max_seq_length": 150,
    "start_token": 1,
    "end_token": 2,
    "pad_token": 0,
}

# Cesty k souborům a adresářům
DEFAULT_OUTPUT_DIR = "outputs"
CHECKPOINT_DIR = "checkpoints"
