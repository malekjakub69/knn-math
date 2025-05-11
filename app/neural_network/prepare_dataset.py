import os
import shutil
import random
from tqdm import tqdm
import argparse
from skimage import io
import six
from .inkml2img import inkml2img


def prepare_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, limit=None):
    """
    Připraví dataset pro neuronovou síť rozdělením na trénovací, validační a testovací sadu.

    Args:
        input_dir: Adresář se vstupními daty a ground truth train_labels.txt
        output_dir: Adresář pro výstupní data
        train_ratio: Poměr trénovacích dat (0.8 = 80%)
        val_ratio: Poměr validačních dat (0.1 = 10%)
        test_ratio: Poměr testovacích dat (0.1 = 10%)
        limit: Maximální počet vzorků (pro testování s menším datasetem)
    """
    # Kontrola, zda je poměr správný
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Poměry pro rozdělení datasetu musí dát dohromady 1.0")

    # Vytvoření adresářů, pokud neexistují
    os.makedirs(os.path.join(output_dir, "train_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val_images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test_images"), exist_ok=True)

    input_images_dir = os.path.join(input_dir, "train_images")
    inkml_dir = os.path.join(input_dir, "inkml")

    # Nacteni seznamu INKML souboru
    # inkml_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(".inkml")]
    inkml_files = [f for f in os.listdir(inkml_dir) if f.lower().endswith(".inkml")]

    # Nacteni existujicich souboru z train_labels.txt
    train_labels_path = os.path.join(input_dir, "train_labels.txt")
    existing_entries = set()

    if os.path.exists(train_labels_path):
        with open(train_labels_path, "r", encoding="utf-8") as f:
            for line in f:
                image_name = line.strip().split("\t")[0]  # Extract the image name
                existing_entries.add(image_name)

    # Pokud existují INKML soubory, vygenerovat obrázky a pridat je do train_labels.txt
    if inkml_files:
        print("Načítání INKML souborů...")
        for inkml_file in inkml_files:
            image_name = os.path.splitext(inkml_file)[0] + ".png"

            # Pokud byl soubor vygenerován dříve, přeskočit
            if image_name in existing_entries:
                # print(f"Soubor {image_name} již existuje v train_labels.txt, přeskočeno.")
                continue

            # inkml_path = os.path.join(input_images_dir, inkml_file)
            inkml_path = os.path.join(inkml_dir, inkml_file)
            # Načtení obrázku z INKML
            image, formula = inkml2img(inkml_path, img_height=80, line_width=1, padding=0, color="black")

            # Uložení obrázku jako PNG
            output_image_path = os.path.join(input_images_dir, image_name)
            io.imsave(output_image_path, image)

            # Přidání do train_labels.txt
            with open(os.path.join(input_dir, "train_labels.txt"), "a", encoding="utf-8") as f:
                f.write(f"{image_name}\t{formula}\n")

    # Načtení seznamu obrázků
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Náhodné zamíchání
    random.shuffle(image_files)

    # Omezení počtu vzorků pro testování
    if limit and limit > 0 and limit < len(image_files):
        image_files = image_files[:limit]

    # Výpočet počtu vzorků pro každou sadu
    num_samples = len(image_files)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    # Rozdělení dat
    train_files = image_files[:num_train]
    val_files = image_files[num_train : num_train + num_val]
    test_files = image_files[num_train + num_val :]

    print(f"Celkem {num_samples} vzorků.")
    print(f"Trénovací sada: {len(train_files)} vzorků")
    print(f"Validační sada: {len(val_files)} vzorků")
    print(f"Testovací sada: {len(test_files)} vzorků")

    # Načtení LaTeX anotací
    labels_file = os.path.join(input_dir, "train_labels.txt")
    labels = {}

    print("Načítání LaTeX anotací...")
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                # Odstraníme případné mezery na začátku a konci
                image_name = parts[0].strip()
                latex_code = parts[1].strip()
                labels[image_name] = latex_code

    # Vytvoření souborů s anotacemi pro každou sadu
    print("Vytváření souborů s anotacemi...")

    # Trénovací sada
    with open(os.path.join(output_dir, "train_formulas.txt"), "w", encoding="utf-8") as f:
        for image_file in train_files:
            if image_file in labels:
                f.write(f"{labels[image_file]}\n")
            else:
                print(f"VAROVÁNÍ: chybí anotace pro {image_file}")

    # Validační sada
    with open(os.path.join(output_dir, "val_formulas.txt"), "w", encoding="utf-8") as f:
        for image_file in val_files:
            if image_file in labels:
                f.write(f"{labels[image_file]}\n")
            else:
                print(f"VAROVÁNÍ: chybí anotace pro {image_file}")

    # Testovací sada
    with open(os.path.join(output_dir, "test_formulas.txt"), "w", encoding="utf-8") as f:
        for image_file in test_files:
            if image_file in labels:
                f.write(f"{labels[image_file]}\n")
            else:
                print(f"VAROVÁNÍ: chybí anotace pro {image_file}")

    # Kopírování souborů
    print("Kopírování obrázků...")

    # Trénovací sada
    print("Kopírování trénovacích dat...")
    for image_file in tqdm(train_files):
        src = os.path.join(input_images_dir, image_file)
        dst = os.path.join(output_dir, "train_images", image_file)
        shutil.copy2(src, dst)

    # Validační sada
    print("Kopírování validačních dat...")
    for image_file in tqdm(val_files):
        src = os.path.join(input_images_dir, image_file)
        dst = os.path.join(output_dir, "val_images", image_file)
        shutil.copy2(src, dst)

    # Testovací sada
    print("Kopírování testovacích dat...")
    for image_file in tqdm(test_files):
        src = os.path.join(input_images_dir, image_file)
        dst = os.path.join(output_dir, "test_images", image_file)
        shutil.copy2(src, dst)

    print("Hotovo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Příprava datasetu pro neuronovou síť")
    parser.add_argument("--input", type=str, required=True, help="Vstupní adresář s datasetem")
    parser.add_argument("--output", type=str, required=True, help="Výstupní adresář pro připravený dataset")
    parser.add_argument("--train", type=float, default=0.8, help="Poměr trénovacích dat (0.8 = 80%)")
    parser.add_argument("--val", type=float, default=0.1, help="Poměr validačních dat (0.1 = 10%)")
    parser.add_argument("--test", type=float, default=0.1, help="Poměr testovacích dat (0.1 = 10%)")
    parser.add_argument("--limit", type=int, default=None, help="Maximální počet vzorků (pro testování)")

    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.train, args.val, args.test, args.limit)
