import os
import cv2
import argparse
from tqdm import tqdm


def is_single_line(image_path, latex_code):
    """
    Kontrola, zda je vzorec jednořádkový.

    Args:
        image_path: Cesta k obrázku
        latex_code: LaTeX kód vzorce

    Returns:
        bool: True pokud je vzorec jednořádkový
    """
    # Kontrola LaTeX kódu
    if "\\\\" in latex_code or "\\begin{array}" in latex_code:
        return False

    # Kontrola poměru stran obrázku
    img = cv2.imread(image_path)
    if img is None:
        return False

    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Typicky jednořádkový vzorec má aspect ratio > 2
    return aspect_ratio > 2


def filter_single_line_formulas(input_dir, output_dir):
    """
    Filtruje pouze jednořádkové vzorce.

    Args:
        input_dir: Vstupní adresář s původním datasetem
        output_dir: Výstupní adresář pro filtrovaná data
    """
    # Vytvoření výstupních adresářů
    os.makedirs(os.path.join(output_dir, "train_images"), exist_ok=True)

    # Načtení anotací
    labels_file = os.path.join(input_dir, "train_labels.txt")
    filtered_labels = []

    print("Filtruji jednořádkové vzorce...")
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue

            image_name, latex_code = parts
            image_path = os.path.join(input_dir, "train_images", image_name)

            if not os.path.exists(image_path):
                continue

            if is_single_line(image_path, latex_code):
                # Kopírování obrázku
                import shutil

                dst_path = os.path.join(output_dir, "train_images", image_name)
                shutil.copy2(image_path, dst_path)
                filtered_labels.append(line)

    # Uložení filtrovaných anotací
    output_labels = os.path.join(output_dir, "train_labels.txt")
    with open(output_labels, "w", encoding="utf-8") as f:
        f.writelines(filtered_labels)

    print(f"Hotovo! Nalezeno {len(filtered_labels)} jednořádkových vzorců.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtrování jednořádkových vzorců")
    parser.add_argument("--input", type=str, required=True, help="Vstupní adresář s datasetem")
    parser.add_argument("--output", type=str, required=True, help="Výstupní adresář pro filtrovaná data")

    args = parser.parse_args()
    filter_single_line_formulas(args.input, args.output)
