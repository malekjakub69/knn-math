#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KNN-Math: Rozpoznávání matematických vzorců
Hlavní modul aplikace
"""

import os
import argparse
import cv2
import numpy as np
from PIL import Image
import torch

from .preprocessing.segmentation import segment_image
from .recognition.knn_classifier import KNNClassifier
from .parsing.latex_converter import convert_to_latex
from .utils.data_loader import load_dataset
from .utils.evaluation import evaluate_results

# Import neuronových sítí
from .neural_network.model import LatexOCRModel
from .neural_network.dataset import create_dataloaders
from .neural_network.train import train_model, evaluate_model

# Import konfigurace
from .config import *


def parse_arguments():
    """Zpracování argumentů příkazové řádky"""
    parser = argparse.ArgumentParser(description="KNN-Math: Rozpoznávání matematických vzorců")
    parser.add_argument("--train", action="store_true", help="Trénování modelu")
    parser.add_argument("--test", action="store_true", help="Testování modelu")
    parser.add_argument("--recognize", type=str, help="Cesta k obrázku k rozpoznání")
    parser.add_argument("--output", type=str, default="output.tex", help="Výstupní soubor")
    parser.add_argument("--mode", choices=["knn", "neural_network", "train_nn", "eval_nn"], default="knn", help="Operation mode")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for NN training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for NN training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    return parser.parse_args()


def process_image_with_knn(image_path, classifier, output_dir=None):
    """
    Zpracování obrázku pomocí KNN klasifikátoru.
    """
    # Načtení a segmentace obrázku
    image = cv2.imread(image_path)
    segments = segment_image(image)

    # Klasifikace segmentů
    classifications = []
    for segment in segments:
        symbol = classifier.classify(segment)
        classifications.append(symbol)

    # Konverze do LaTeX
    latex_expression = convert_to_latex(classifications)

    # Uložení výstupu, pokud je specifikován výstupní adresář
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Uložení LaTeX výrazu
        output_path = os.path.join(output_dir, os.path.basename(image_path) + ".tex")
        with open(output_path, "w") as f:
            f.write(latex_expression)

    return latex_expression


def process_image_with_neural_network(image_path, model, device, output_dir=None):
    """
    Zpracování obrázku pomocí neuronové sítě.
    """
    from torchvision import transforms

    # Předzpracování obrázku
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Načtení obrázku
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predikce LaTeX kódu
    model.eval()
    with torch.no_grad():
        predicted_indices = model.predict(image_tensor)

    # Převod indexů na LaTeX tokeny
    # Poznámka: tady musíme mít přístup ke slovníku
    # Pro jednoduchost předpokládejme, že model ho má jako atribut
    idx2word = getattr(model, "idx2word", None)
    if idx2word is None:
        raise ValueError("Model doesn't have idx2word attribute. Load it separately.")

    # Rekonstrukce LaTeX výrazu
    latex_tokens = [idx2word[idx] for idx in predicted_indices if idx not in [0, 1, 2]]  # Vynechání speciálních tokenů
    latex_expression = "".join(latex_tokens)

    # Uložení výstupu, pokud je specifikován výstupní adresář
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Uložení LaTeX výrazu
        output_path = os.path.join(output_dir, os.path.basename(image_path) + ".tex")
        with open(output_path, "w") as f:
            f.write(latex_expression)

    return latex_expression


def main():
    """Hlavní funkce aplikace"""
    args = parse_arguments()

    if args.train:
        train_model()

    if args.test:
        test_model()

    if args.recognize:
        recognize_image(args.recognize, args.output)

    if args.mode == "knn":
        if not args.image:
            parser.error("--image is required when mode is 'knn'")

        if not args.dataset:
            parser.error("--dataset is required to load the KNN model")

        # Načtení datasetu a trénování KNN
        train_data = load_dataset(args.dataset)
        classifier = KNNClassifier(k=3)
        classifier.train(train_data)

        # Zpracování obrázku
        latex_expression = process_image_with_knn(args.image, classifier, args.output)
        print(f"Generated LaTeX: {latex_expression}")

    elif args.mode == "neural_network":
        if not args.image:
            parser.error("--image is required when mode is 'neural_network'")

        if not args.checkpoint:
            parser.error("--checkpoint is required to load the neural network model")

        # Nastavení zařízení (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Načtení modelu
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Tady předpokládáme, že známe vocab_size, v reálném případě by bylo třeba ho načíst
        model = LatexOCRModel(vocab_size=1000)  # Dummy value, replace with actual vocab size
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        # Zpracování obrázku
        latex_expression = process_image_with_neural_network(args.image, model, device, args.output)
        print(f"Generated LaTeX: {latex_expression}")

    elif args.mode == "train_nn":
        if not args.dataset:
            parser.error("--dataset is required for training")

        # Nastavení zařízení (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vytvoření dataloaderu
        train_loader, val_loader, test_loader, vocab_size = create_dataloaders(args.dataset, batch_size=args.batch_size)

        # Vytvoření nebo načtení modelu
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model = LatexOCRModel(vocab_size=vocab_size)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model = LatexOCRModel(vocab_size=vocab_size)

        # Trénování modelu
        train_model(model=model, train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, device=device, checkpoint_path=args.output)

    elif args.mode == "eval_nn":
        if not args.dataset:
            parser.error("--dataset is required for evaluation")

        if not args.checkpoint:
            parser.error("--checkpoint is required to load the model")

        # Nastavení zařízení (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Vytvoření dataloaderu
        train_loader, val_loader, test_loader, vocab_size = create_dataloaders(args.dataset, batch_size=args.batch_size)

        # Načtení modelu
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = LatexOCRModel(vocab_size=vocab_size)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Evaluace modelu
        test_loss, accuracy, seq_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device)

        print(f"Test Loss: {test_loss:.4f}, Token Accuracy: {accuracy:.4f}, Sequence Accuracy: {seq_accuracy:.4f}")

    if not (args.train or args.test or args.recognize or args.mode):
        print("Nebyla zadána žádná akce. Použijte --help pro nápovědu.")


if __name__ == "__main__":
    main()
