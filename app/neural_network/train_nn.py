"""
Trénovací skript pro neuronovou síť.
"""

import os
import argparse
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm

from .model import LatexOCRModel
from .dataset import create_dataloaders, LatexDataset
from .train import train_model, evaluate_model


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)  # You can parameterize the seed if needed

    parser = argparse.ArgumentParser(description="Trénování modelu pro převod matematických výrazů do LaTeX")
    # Data a výstupy
    parser.add_argument("--data_dir", type=str, required=True, help="Adresář s datasetem")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Adresář pro výstupy")
    # Parametry tréninku
    parser.add_argument("--batch_size", type=int, default=64, help="Velikost batch")
    parser.add_argument("--epochs", type=int, default=100, help="Počet epoch")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Cesta k checkpointu modelu")
    parser.add_argument("--eval_only", action="store_true", help="Pouze vyhodnocení modelu bez tréninku")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "mps", "cuda", "auto"], help="Zařízení pro trénink - cpu, mps (pro Apple Silicon), cuda (nVidia GPU) nebo auto (automatická detekce)")
    # Parametry modelu
    parser.add_argument("--encoder_dim", type=int, default=320, help="Dimenze encoderu")
    parser.add_argument("--num_transformer_layers", type=int, default=4, help="Počet transformer vrstev")
    parser.add_argument("--decoder_dim", type=int, default=512, help="Dimenze decoderu")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimenze embeddingu")
    parser.add_argument("--attention_dim", type=int, default=256, help="Dimenze attention")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")    # Augmentace
    parser.add_argument("--no_augment", action="store_true", help="Vypnout datovou augmentaci")

    args = parser.parse_args()

    # Vytvoření output adresáře, pokud neexistuje
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Vytvoření adresáře pro checkpointy
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Nastavení zařízení (CPU/MPS/CUDA)
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Používám akceleraci Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Používám CUDA (GPU: {torch.cuda.get_device_name(0)})")
        else:
            device = torch.device("cpu")
            print("Používám CPU (MPS ani CUDA nejsou dostupné)")
    else:
        device = torch.device(args.device)
        if args.device == "mps" and not torch.backends.mps.is_available():
            print("VAROVÁNÍ: Požadujete MPS, ale není dostupné. Přepínám na CPU.")
            device = torch.device("cpu")
        elif args.device == "cuda" and not torch.cuda.is_available():
            print("VAROVÁNÍ: Požadujete CUDA, ale není dostupné. Přepínám na CPU.")
            device = torch.device("cpu")

    print(f"Používané zařízení: {device}")

    # Vytvoření dataloaderu
    augment = not args.no_augment
    train_loader, val_loader, test_loader, vocab_size = create_dataloaders(args.data_dir, batch_size=args.batch_size, augment=augment)

    # Uložení slovníku pro pozdější použití při predikci
    train_dataset = LatexDataset(args.data_dir, split="train")
    vocab_file = os.path.join(args.output_dir, "idx2word.pkl")
    with open(vocab_file, "wb") as f:
        pickle.dump(train_dataset.idx2word, f)
    print(f"Slovník uložen do: {vocab_file}")

    # Parametry modelu
    model_params = {"encoder_dim": args.encoder_dim, "decoder_dim": args.decoder_dim, "embedding_dim": args.embedding_dim, "attention_dim": args.attention_dim, "vocab_size": vocab_size, "dropout": args.dropout}

    # Vytvoření nebo načtení modelu
    if args.checkpoint:
        print(f"Načítání modelu z checkpointu: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = LatexOCRModel(**model_params)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Vytváření nového modelu")
        print(f"Parametry modelu: {model_params}")
        model = LatexOCRModel(**model_params)

    # Přesun modelu na zvolené zařízení
    model = model.to(device)

    if args.eval_only:
        # Pouze vyhodnocení modelu
        print("Vyhodnocování modelu...")
        test_loss, accuracy = evaluate_model(model=model, test_loader=test_loader, device=device)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    else:
        # Trénování modelu
        print("Spouštění tréninku...")
        model = train_model(model=model, train_loader=train_loader, val_loader=val_loader, learning_rate=args.learning_rate, epochs=args.epochs, device=device, checkpoint_path=checkpoint_dir)
        # Vyhodnocení natrénovaného modelu
        print("Vyhodnocování modelu...")
        test_loss, accuracy = evaluate_model(model=model, test_loader=test_loader, device=device)
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
