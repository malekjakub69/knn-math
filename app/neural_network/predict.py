import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import pickle
from .model import LatexOCRModel


def predict_single_image(model, image_path, idx2word, max_length=150, start_token=1, end_token=2, device="cpu", beam_size=3):
    """
    Predikce LaTeX kódu pro jeden obrázek.

    Args:
        model: Natrénovaný model
        image_path: Cesta k obrázku
        idx2word: Slovník pro převod indexů na tokeny
        max_length: Maximální délka generované sekvence
        start_token: Index start tokenu
        end_token: Index end tokenu
        device: Zařízení pro výpočet (mps/cpu)
        beam_size: Velikost beam search (počet kandidátů)

    Returns:
        Predikovaný LaTeX kód
    """
    # Načtení a předzpracování obrázku
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predikce
    model.eval()
    with torch.no_grad():
        predicted_indices = model.predict(image, max_length, start_token, end_token, beam_size=beam_size)

    # Převod indexů na LaTeX tokeny
    latex_tokens = [idx2word[idx] for idx in predicted_indices if idx not in [0, 1, 2]]  # Vynechání speciálních tokenů

    # Post-processing pro odstranění chyb
    latex_expression = post_process_latex("".join(latex_tokens))

    return latex_expression


def post_process_latex(latex_str):
    """
    Post-processing LaTeX výrazu pro zvýšení kvality.
    Opraví běžné chyby a zajistí konzistenci závorek a dalších symbolů.
    """
    # Oprava nekonzistentních závorek
    open_brackets = latex_str.count("{")
    close_brackets = latex_str.count("}")

    if open_brackets > close_brackets:
        latex_str += "}" * (open_brackets - close_brackets)
    elif close_brackets > open_brackets:
        # Odstraníme přebývající }
        excess = close_brackets - open_brackets
        for _ in range(excess):
            last_idx = latex_str.rfind("}")
            if last_idx != -1:
                latex_str = latex_str[:last_idx] + latex_str[last_idx + 1 :]

    # Oprava nekonzistentních \left \right
    left_cmds = latex_str.count("\\left")
    right_cmds = latex_str.count("\\right")

    if left_cmds > right_cmds:
        latex_str += "\\right." * (left_cmds - right_cmds)

    # Odstranění opakujících se mezer
    while "  " in latex_str:
        latex_str = latex_str.replace("  ", " ")

    # Odstranění mezer na začátku a konci
    latex_str = latex_str.strip()

    return latex_str


def main():
    parser = argparse.ArgumentParser(description="Predikce LaTeX kódu pro obrázek matematického výrazu")
    parser.add_argument("--checkpoint", type=str, default="outputs/neural_network/checkpoints/best_model.pth", help="Cesta k checkpointu modelu")
    parser.add_argument("--vocab", type=str, default="outputs/neural_network/idx2word.pkl", help="Cesta k souboru se slovníkem")
    parser.add_argument("--image", type=str, required=True, help="Cesta k obrázku")
    parser.add_argument("--output", type=str, default=None, help="Cesta k výstupnímu souboru")
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "mps", "auto"], help="Zařízení pro predikci - cpu, mps (pro Apple Silicon) nebo auto (automatická detekce)")
    parser.add_argument("--beam_size", type=int, default=3, help="Velikost beam search (počet kandidátů)")

    args = parser.parse_args()

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

    # Načtení slovníku
    print(f"Načítání slovníku z: {args.vocab}")
    with open(args.vocab, "rb") as f:
        idx2word = pickle.load(f)

    vocab_size = len(idx2word)
    print(f"Velikost slovníku: {vocab_size}")

    # Načtení modelu
    print(f"Načítání modelu z: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = LatexOCRModel(vocab_size=vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    # Predikce
    print(f"Predikce pro obrázek: {args.image}")
    latex_expression = predict_single_image(model, args.image, idx2word, device=device, beam_size=args.beam_size)

    print(f"Predikovaný LaTeX: {latex_expression}")

    # Uložení výsledku
    if args.output:
        with open(args.output, "w") as f:
            f.write(latex_expression)
        print(f"Výsledek uložen do: {args.output}")

    return latex_expression


if __name__ == "__main__":
    main()
