import os
import argparse
import torch
import pickle
from .model import LatexOCRModel
from .dataset import create_dataloaders, LatexDataset
from .train import train_model, evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Trénování modelu pro převod matematických výrazů do LaTeX")
    parser.add_argument("--data_dir", type=str, required=True, help="Adresář s datasetem")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Adresář pro výstupy")
    parser.add_argument("--batch_size", type=int, default=32, help="Velikost batch")
    parser.add_argument("--epochs", type=int, default=30, help="Počet epoch")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Cesta k checkpointu modelu")
    parser.add_argument("--eval_only", action="store_true", help="Pouze vyhodnocení modelu bez tréninku")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "mps", "auto"], help="Zařízení pro trénink - cpu, mps (pro Apple Silicon) nebo auto (automatická detekce)")

    args = parser.parse_args()

    # Vytvoření output adresáře, pokud neexistuje
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Vytvoření adresáře pro checkpointy
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Nastavení zařízení (CPU/MPS)
    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Používám akceleraci Apple Silicon (MPS)")
        else:
            device = torch.device("cpu")
            print("Používám CPU (MPS není dostupné)")
    else:
        device = torch.device(args.device)
        if args.device == "mps" and not torch.backends.mps.is_available():
            print("VAROVÁNÍ: Požadujete MPS, ale není dostupné. Přepínám na CPU.")
            device = torch.device("cpu")

    print(f"Používané zařízení: {device}")

    # Vytvoření dataloaderu
    train_loader, val_loader, test_loader, vocab_size = create_dataloaders(args.data_dir, batch_size=args.batch_size)

    # Uložení slovníku pro pozdější použití při predikci
    train_dataset = LatexDataset(args.data_dir, split="train")
    vocab_file = os.path.join(args.output_dir, "idx2word.pkl")
    with open(vocab_file, "wb") as f:
        pickle.dump(train_dataset.idx2word, f)
    print(f"Slovník uložen do: {vocab_file}")

    # Vytvoření nebo načtení modelu
    if args.checkpoint:
        print(f"Načítání modelu z checkpointu: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = LatexOCRModel(vocab_size=vocab_size)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Vytváření nového modelu")
        model = LatexOCRModel(vocab_size=vocab_size)

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
