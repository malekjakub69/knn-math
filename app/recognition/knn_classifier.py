"""
Implementace KNN klasifikátoru pro rozpoznávání matematických symbolů
"""

import os
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from ..config import KNN_N_NEIGHBORS, KNN_WEIGHTS


def train(X_train, y_train):
    """
    Trénování KNN klasifikátoru

    Args:
        X_train: Trénovací data
        y_train: Trénovací štítky

    Returns:
        Natrénovaný model
    """
    # Převod 2D obrázků na 1D vektory
    X_train_flat = [img.flatten() for img in X_train]

    # Inicializace a trénování klasifikátoru
    knn = KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS)
    knn.fit(X_train_flat, y_train)

    return knn


def predict(model, image):
    """
    Predikce symbolu na základě jednoho obrázku

    Args:
        model: Natrénovaný KNN model
        image: Předzpracovaný obrázek symbolu

    Returns:
        Predikovaný symbol
    """
    # Převod 2D obrázku na 1D vektor
    image_flat = image.flatten().reshape(1, -1)

    # Predikce
    return model.predict(image_flat)[0]


def save_model(model, path):
    """
    Uložení natrénovaného modelu

    Args:
        model: Natrénovaný model
        path: Cesta pro uložení modelu
    """
    # Vytvoření adresáře, pokud neexistuje
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Uložení modelu pomocí pickle
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """
    Načtení natrénovaného modelu

    Args:
        path: Cesta k uloženému modelu

    Returns:
        Načtený model
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
