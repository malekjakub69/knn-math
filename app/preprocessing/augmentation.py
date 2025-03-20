import numpy as np
import cv2
import random
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps, ImageEnhance


class AugmentationPipeline:
    """
    Pipeline pro augmentaci obrázků matematických rovnic
    """

    def __init__(self, prob=0.5):
        """
        Inicializace augmentačního pipeline.

        Args:
            prob: Pravděpodobnost aplikace každé augmentace
        """
        self.prob = prob

    def __call__(self, image):
        """
        Aplikace náhodných augmentací na obrázek.

        Args:
            image: PIL Image nebo tensor

        Returns:
            Augmentovaný obrázek stejného typu jako vstup
        """
        # Převod na PIL Image pokud je vstup tensor
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            if image.dim() == 3 and image.shape[0] == 3:  # CHW format
                # Denormalizace pokud je tensor normalizovaný
                image = image.permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
                image = Image.fromarray(image)
            else:
                image = transforms.ToPILImage()(image)

        # Aplikace náhodných augmentací

        # 1. Rotace (mírná)
        if random.random() < self.prob:
            angle = random.uniform(-5, 5)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)

        # 2. Změna měřítka
        if random.random() < self.prob:
            scale = random.uniform(0.8, 1.2)
            width, height = image.size
            new_width, new_height = int(width * scale), int(height * scale)
            image = image.resize((new_width, new_height), Image.BILINEAR)

            # Úprava velikosti na původní rozměry
            result = Image.new(image.mode, (width, height), (255, 255, 255))
            paste_x = max(0, (width - new_width) // 2)
            paste_y = max(0, (height - new_height) // 2)
            result.paste(image, (paste_x, paste_y))
            image = result

        # 3. Změna jasu a kontrastu
        if random.random() < self.prob:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # 4. Přidání šumu
        if random.random() < self.prob:
            # Konverze na numpy array
            img_array = np.array(image)

            # Gaussian noise
            noise = np.random.normal(0, 5, img_array.shape)
            noisy_img = img_array + noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

            image = Image.fromarray(noisy_img)

        # 5. Mírné rozmazání
        if random.random() < self.prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))

        # 6. Perspektivní transformace (mírná)
        if random.random() < self.prob:
            width, height = image.size

            # Náhodné posunutí rohů (mírné)
            shift = width * 0.05

            # Zdrojové body (původní rohy)
            src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # Levý horní  # Pravý horní  # Levý dolní  # Pravý dolní

            # Cílové body (posunuté rohy)
            dst_points = np.float32(
                [
                    [random.uniform(0, shift), random.uniform(0, shift)],  # Levý horní
                    [width - random.uniform(0, shift), random.uniform(0, shift)],  # Pravý horní
                    [random.uniform(0, shift), height - random.uniform(0, shift)],  # Levý dolní
                    [width - random.uniform(0, shift), height - random.uniform(0, shift)],  # Pravý dolní
                ]
            )
            # Výpočet transformační matice
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Konverze PIL na OpenCV formát
            img_cv = np.array(image)

            # Aplikace perspektivní transformace
            transformed_img = cv2.warpPerspective(img_cv, transform_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            # Konverze zpět na PIL
            image = Image.fromarray(transformed_img)

        # 7. Drobné elastic deformace (pro simulaci ručně psaných rovnic)
        if random.random() < self.prob:
            img_cv = np.array(image)

            # Parametry deformace
            alpha = random.uniform(10, 20)  # Síla deformace
            sigma = random.uniform(3, 5)  # Elasticita

            # Vytvoření náhodných displacement fieldů
            shape = img_cv.shape[:2]
            dx = np.random.rand(*shape) * 2 - 1
            dy = np.random.rand(*shape) * 2 - 1

            # Gaussian filtrace pro plynulé deformace
            dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

            # Aplikace deformace
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            map_x = np.float32(x + dx)
            map_y = np.float32(y + dy)

            warped = cv2.remap(img_cv, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            image = Image.fromarray(warped)

        # Převod zpět na tensor pokud byl vstup tensor
        if is_tensor:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            image = transform(image)

        return image


def create_augmented_dataset(original_dataset, augmentation_factor=3):
    """
    Vytvoří augmentovanou verzi datasetu.

    Args:
        original_dataset: Původní dataset (seznam dvojic (obrázek, LaTeX))
        augmentation_factor: Kolikrát zvětšit dataset pomocí augmentace

    Returns:
        Rozšířený dataset
    """
    augmenter = AugmentationPipeline(prob=0.5)
    augmented_dataset = []

    # Přidání původních dat
    augmented_dataset.extend(original_dataset)

    # Přidání augmentovaných dat
    for _ in range(augmentation_factor - 1):
        for image, latex in original_dataset:
            # Augmentace obrázku
            augmented_image = augmenter(image)
            # Přidání do rozšířeného datasetu
            augmented_dataset.append((augmented_image, latex))

    return augmented_dataset


def demo_augmentations(image_path, output_dir, num_samples=5):
    """
    Demonstrace různých augmentací na jednom obrázku.

    Args:
        image_path: Cesta k obrázku
        output_dir: Adresář pro uložení augmentovaných obrázků
        num_samples: Počet vzorků pro každou augmentaci
    """
    import os

    # Načtení obrázku
    image = Image.open(image_path).convert("RGB")

    # Vytvoření augmentera s 100% pravděpodobností pro demonstraci
    augmenter = AugmentationPipeline(prob=1.0)

    # Uložení původního obrázku
    image.save(os.path.join(output_dir, "original.png"))

    # Vygenerování a uložení augmentovaných vzorků
    for i in range(num_samples):
        augmented = augmenter(image)
        augmented.save(os.path.join(output_dir, f"augmented_{i+1}.png"))
