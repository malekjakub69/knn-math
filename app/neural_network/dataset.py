import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random


class GaussianNoise:
    """Custom transform to add Gaussian noise to tensor images"""
    def __init__(self, std=0.02):
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0, 1)  # Ensure values stay in [0,1]


class LatexDataset(Dataset):
    """
    Dataset pro obrázky matematických výrazů a jejich LaTeX kód.
    """

    def __init__(self, data_dir, split="train", max_length=150, transform=None, augment=True):
        """
        Inicializace datasetu.

        Args:
            data_dir: Adresář s daty
            split: 'train', 'val' nebo 'test'
            max_length: Maximální délka LaTeX sekvence
            transform: Transformace aplikované na obrázky
            augment: Zda použít augmentaci (pouze pro trénovací data)
        """
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.augment = augment and split == "train"  # Augmentaci používáme jen na trénovací data

        # Transformace pro různé fáze
        if transform:
            self.transform = transform
        else:
            # Standartní normalizace
            norm_mean = [0.485, 0.456, 0.406]
            norm_std = [0.229, 0.224, 0.225]

            # Základní transformace - použité pro všechny
            base_transform = [
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ]

            # Augmentace pouze pro trénink
            if self.augment:
                color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
                blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                
                self.transform = transforms.Compose([
                    transforms.RandomApply([color_jitter], p=0.4),
                    transforms.RandomApply([blur], p=0.3),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomAffine(degrees=2, translate=(0.12, 0.12), scale=(0.95, 1.05)),
                    *base_transform,
                    transforms.RandomApply([GaussianNoise(0.02)], p=0.2),
                    transforms.RandomErasing(p=0.2),
                ])
            else:
                self.transform = transforms.Compose(base_transform)

        # Načtení seznamu obrázků a odpovídajících LaTeX kódů
        self.images_path = os.path.join(data_dir, f"{split}_images")
        self.formulas_file = os.path.join(data_dir, f"{split}_formulas.txt")

        # Načtení seznamu obrázků
        self.image_files = sorted([f for f in os.listdir(self.images_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

        # Načtení LaTeX formulí
        with open(self.formulas_file, "r", encoding="utf-8") as f:
            self.formulas = [line.strip() for line in f.readlines()]

        # Kontrola, že počet obrázků odpovídá počtu formulí
        assert len(self.image_files) == len(self.formulas), f"Počet obrázků ({len(self.image_files)}) a formulí ({len(self.formulas)}) se neshoduje"

        # Vytvoření tokenizeru pro LaTeX
        self._create_vocabulary()

    def _create_vocabulary(self):
        """
        Vytvoření slovníku pro tokenizaci LaTeX kódu.
        """
        # Speciální tokeny
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}

        # Vytvoření slovníku ze všech formulí v trénovacím setu
        if self.split == "train":
            # Tokenizace založená na znacích a LaTeX příkazech
            unique_tokens = set()
            for formula in self.formulas:
                # Jednoduchá tokenizace - lze vylepšit pro lepší zachycení LaTeX struktury
                i = 0
                while i < len(formula):
                    if formula[i] == "\\":
                        # LaTeX příkaz
                        j = i + 1
                        while j < len(formula) and formula[j].isalpha():
                            j += 1
                        token = formula[i:j]
                        unique_tokens.add(token)
                        i = j
                    else:
                        # Jednotlivý znak
                        unique_tokens.add(formula[i])
                        i += 1

            # Přidání tokenů do slovníku
            for i, token in enumerate(sorted(unique_tokens)):
                self.word2idx[token] = i + 4  # +4 kvůli speciálním tokenům
                self.idx2word[i + 4] = token

            # Uložení velikosti slovníku
            self.vocab_size = len(self.word2idx)

            print(f"Vytvořen slovník s {self.vocab_size} tokeny.")

    def tokenize(self, formula):
        """
        Převod LaTeX formule na sekvenci tokenů.
        """
        tokens = []
        i = 0
        while i < len(formula):
            if formula[i] == "\\":
                # LaTeX příkaz
                j = i + 1
                while j < len(formula) and formula[j].isalpha():
                    j += 1
                token = formula[i:j]
                tokens.append(self.word2idx.get(token, self.word2idx["<UNK>"]))
                i = j
            else:
                # Jednotlivý znak
                tokens.append(self.word2idx.get(formula[i], self.word2idx["<UNK>"]))
                i += 1

        return tokens

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Načtení obrázku a příslušné formule.
        """
        # Načtení obrázku
        image_path = os.path.join(self.images_path, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        # Pre-processing - vylepšený kontrast pro lepší rozpoznání matematických symbolů
        if self.split == "train" and random.random() < 0.3:
            # Pro některé trénovací obrázky zvýšíme kontrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)

        if self.transform:
            image = self.transform(image)

        # Načtení a tokenizace formule
        formula = self.formulas[idx]

        # Tokenizace
        tokens = [self.word2idx["<START>"]]
        tokens.extend(self.tokenize(formula))
        tokens.append(self.word2idx["<END>"])

        # Omezení délky
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.word2idx["<END>"]]

        # Převod na tensor
        caption = torch.LongTensor(tokens)
        caption_length = len(tokens)

        return image, caption, caption_length


# Definice collate function pro padding sekvencí
def collate_fn(batch):
    # Rozdělení batch na obrázky, caption a délky
    images, captions, lengths = zip(*batch)

    # Stacking obrázků
    images = torch.stack(images, 0)

    # Nalezení maximální délky v tomto batch
    max_length = max(lengths)

    # Vytvoření tensoru pro padded captions
    padded_captions = torch.zeros(len(captions), max_length).long()

    # Padding všech caption na max_length
    for i, cap in enumerate(captions):
        end = lengths[i]
        padded_captions[i, :end] = cap[:end]

    return images, padded_captions, torch.LongTensor(lengths)


def create_dataloaders(data_dir, batch_size=32, num_workers=0, augment=True):
    """
    Vytvoření dataloaderu pro train, val a test data.
    """
    # Vytvoření datasetů
    print(f"Načítání datasetu z: {data_dir}")
    train_dataset = LatexDataset(data_dir, split="train", augment=augment)
    val_dataset = LatexDataset(data_dir, split="val", augment=False)
    test_dataset = LatexDataset(data_dir, split="test", augment=False)

    print(f"Trénovací dataset: {len(train_dataset)} vzorků")
    print(f"Validační dataset: {len(val_dataset)} vzorků")
    print(f"Testovací dataset: {len(test_dataset)} vzorků")

    # Vytvoření dataloaderu
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, train_dataset.vocab_size
