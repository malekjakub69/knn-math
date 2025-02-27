"""
Funkce pro vyhodnocení přesnosti modelu
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Vyhodnocení přesnosti modelu

    Args:
        model: Natrénovaný model
        X_test: Testovací data
        y_test: Testovací štítky

    Returns:
        Přesnost modelu v procentech
    """
    # Převod 2D obrázků na 1D vektory
    X_test_flat = [img.flatten() for img in X_test]

    # Predikce
    y_pred = model.predict(X_test_flat)

    # Výpočet přesnosti
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Tisk detailní zprávy o klasifikaci
    print("Klasifikační zpráva:")
    print(classification_report(y_test, y_pred))

    # Výpočet matice záměn (pro detailnější analýzu)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Matice záměn:")
    print(conf_matrix)

    return accuracy
