"""
Modul pro segmentaci a předzpracování matematických symbolů
"""

import cv2
import numpy as np
# from ..config import IMAGE_SIZE
from config import IMAGE_SIZE


def preprocess_image(image):
    """
    Předzpracování jednoho obrázku

    Args:
        image: Vstupní obrázek

    Returns:
        Předzpracovaný obrázek vhodný pro klasifikátor
    """
    # Převod na stupně šedi, pokud je barevný
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Prahování pro oddělení pozadí a textu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Změna velikosti na standardní rozměr
    resized = cv2.resize(binary, IMAGE_SIZE)

    # Normalizace hodnot pixelů
    normalized = resized / 255.0

    return normalized


def preprocess_images(images):
    """
    Předzpracování sady obrázků

    Args:
        images: Seznam vstupních obrázků

    Returns:
        Seznam předzpracovaných obrázků
    """
    return [preprocess_image(img) for img in images]


def segment_symbols(image):
    """
    Segmentace matematických symbolů z celého obrázku

    Args:
        image: Vstupní obrázek s matematickým výrazem

    Returns:
        Seznam obrázků jednotlivých symbolů
    """
    # Převod na stupně šedi
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Prahování
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Hledání kontur symbolů
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extrakce symbolů
    symbols = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Filtrování příliš malých oblastí (šum)
        if w > 5 and h > 5:
            symbol = gray[y : y + h, x : x + w]
            symbols.append(symbol)

    # Seřazení symbolů zleva doprava
    symbols.sort(key=lambda x: cv2.boundingRect(cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])[0])

    return symbols
