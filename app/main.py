#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KNN-Math: Rozpoznávání matematických vzorců
Hlavní modul aplikace
"""

import os
import argparse
from preprocessing import segmentation
from recognition import knn_classifier
from parsing import latex_converter
from utils import data_loader, evaluation
import config


def parse_arguments():
    """Zpracování argumentů příkazové řádky"""
    parser = argparse.ArgumentParser(description="KNN-Math: Rozpoznávání matematických vzorců")
    parser.add_argument("--train", action="store_true", help="Trénování modelu")
    parser.add_argument("--test", action="store_true", help="Testování modelu")
    parser.add_argument("--recognize", type=str, help="Cesta k obrázku k rozpoznání")
    parser.add_argument("--output", type=str, default="output.tex", help="Výstupní soubor")
    return parser.parse_args()


def train_model():
    """Trénování KNN modelu"""
    print("Načítání trénovacích dat...")
    X_train, y_train = data_loader.load_training_data(os.path.join(config.DATASET_PATH, "train_labels.txt"), config.DATASET_PATH)

    print("Předzpracování trénovacích dat...")
    X_train_processed = segmentation.preprocess_images(X_train)

    print("Trénování KNN klasifikátoru...")
    model = knn_classifier.train(X_train_processed, y_train)
    knn_classifier.save_model(model, config.MODEL_PATH)

    print(f"Model úspěšně uložen do {config.MODEL_PATH}")


def test_model():
    """Testování KNN modelu"""
    print("Načítání testovacích dat...")
    X_test, y_test = data_loader.load_test_data(os.path.join(config.DATASET_PATH, "test_labels.txt"), config.DATASET_PATH)

    print("Předzpracování testovacích dat...")
    X_test_processed = segmentation.preprocess_images(X_test)

    print("Načítání modelu...")
    model = knn_classifier.load_model(config.MODEL_PATH)

    print("Vyhodnocení přesnosti...")
    accuracy = evaluation.evaluate_model(model, X_test_processed, y_test)
    print(f"Přesnost modelu: {accuracy:.2f}%")


def recognize_image(image_path, output_path):
    """Rozpoznání matematického vzorce v obrázku"""
    print(f"Rozpoznávání vzorce v obrázku: {image_path}")

    # Načtení obrázku
    image = data_loader.load_image(image_path)

    # Segmentace symbolů
    symbols = segmentation.segment_symbols(image)

    # Načtení modelu
    model = knn_classifier.load_model(config.MODEL_PATH)

    # Rozpoznání jednotlivých symbolů
    recognized_symbols = []
    for symbol_img in symbols:
        processed_img = segmentation.preprocess_image(symbol_img)
        recognized = knn_classifier.predict(model, processed_img)
        recognized_symbols.append(recognized)

    # Převod do LaTeXu
    latex_code = latex_converter.convert_to_latex(recognized_symbols)

    # Uložení výsledku
    with open(output_path, "w") as f:
        f.write(latex_code)

    print(f"Rozpoznaný vzorec uložen do: {output_path}")
    print(f"LaTeX kód: {latex_code}")


def main():
    """Hlavní funkce aplikace"""
    args = parse_arguments()

    if args.train:
        train_model()

    if args.test:
        test_model()

    if args.recognize:
        recognize_image(args.recognize, args.output)

    if not (args.train or args.test or args.recognize):
        print("Nebyla zadána žádná akce. Použijte --help pro nápovědu.")


if __name__ == "__main__":
    main()
