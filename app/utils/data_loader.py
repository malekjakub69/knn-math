"""
Funkce pro načítání a zpracování dat
"""

import os
import cv2
import numpy as np


def load_image(path):
    """
    Načte obrázek z cesty

    Args:
        path: Cesta k obrázku

    Returns:
        Načtený obrázek
    """
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def load_training_data(labels_path, images_dir):
    """
    Načte trénovací data z adresáře

    Args:
        labels_path: Cesta k souboru s popisky
        images_dir: Adresář s obrázky

    Returns:
        X_train: Seznam obrázků
        y_train: Seznam popisků
    """
    X_train = []
    y_train = []

    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_path = os.path.join(images_dir, parts[0])
                label = parts[1]

                if os.path.exists(image_path):
                    image = load_image(image_path)
                    if image is not None:
                        X_train.append(image)
                        y_train.append(label)

    return X_train, y_train


def load_test_data(labels_path, images_dir):
    """
    Načte testovací data z adresáře

    Args:
        labels_path: Cesta k souboru s popisky
        images_dir: Adresář s obrázky

    Returns:
        X_test: Seznam obrázků
        y_test: Seznam popisků
    """
    # Pro jednoduchost používáme stejnou implementaci jako pro trénovací data
    return load_training_data(labels_path, images_dir)
