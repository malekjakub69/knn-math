import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm


def train_model(model, train_loader, val_loader, learning_rate=3e-4, epochs=30, device="cuda" if torch.cuda.is_available() else "cpu", checkpoint_path="checkpoints"):
    """
    Trénink modelu pro převod matematických výrazů do LaTeX.
    """
    # Přesun modelu na zvolené zařízení (GPU/CPU)
    model = model.to(device)

    # Definice loss funkce a optimizeru
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    # Nejlepší validační loss
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Trénink
        model.train()
        train_loss = 0.0
        start_time = time.time()

        # Průchod trénovacím datasetem
        for i, (images, captions, lengths) in enumerate(tqdm(train_loader)):
            # Přesun na zvolené zařízení
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(images, captions, lengths)

            # Reshape pro výpočet loss
            outputs = outputs.view(-1, outputs.size(2))
            targets = captions.view(-1)

            # Výpočet loss
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Optimalizace
            optimizer.step()

            # Akumulace loss
            train_loss += loss.item()

            # Log každých 100 batch
            if (i + 1) % 100 == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Průměrná tréninková loss
        avg_train_loss = train_loss / len(train_loader)

        # Validace
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, captions, lengths in tqdm(val_loader):
                # Přesun na zvolené zařízení
                images = images.to(device)
                captions = captions.to(device)

                # Forward pass
                outputs = model(images, captions, lengths)

                # Reshape pro výpočet loss
                outputs = outputs.view(-1, outputs.size(2))
                targets = captions.view(-1)

                # Výpočet loss
                loss = criterion(outputs, targets)

                # Akumulace loss
                val_loss += loss.item()

        # Průměrná validační loss
        avg_val_loss = val_loss / len(val_loader)

        # Learning rate scheduler
        scheduler.step(avg_val_loss)

        # Výpis statistik
        time_elapsed = time.time() - start_time
        print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(f"Time: {time_elapsed:.2f}s")

        # Uložení nejlepšího modelu
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                },
                f"{checkpoint_path}/best_model.pth",
            )
            print("Model saved!")

        # Uložení checkpointu každých 5 epoch
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                },
                f"{checkpoint_path}/model_epoch_{epoch+1}.pth",
            )

    print("Training complete!")
    return model


def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Vyhodnocení natrénovaného modelu na testovacím datasetu.
    """
    # Přesun modelu na zvolené zařízení
    model = model.to(device)
    model.eval()

    # Loss funkce
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Metriky
    test_loss = 0.0
    correct_predictions = 0
    total_tokens = 0

    with torch.no_grad():
        for images, captions, lengths in tqdm(test_loader):
            # Přesun na zvolené zařízení
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(images, captions, lengths)

            # Reshape pro výpočet loss
            batch_size = outputs.size(0)
            outputs_flat = outputs.view(-1, outputs.size(2))
            targets_flat = captions.view(-1)

            # Výpočet loss
            loss = criterion(outputs_flat, targets_flat)
            test_loss += loss.item()

            # Výpočet accuracy
            _, predictions = outputs.max(2)

            # Maska pro ignorování padding tokenů
            mask = captions != 0

            # Počet správných predikcí
            correct = (predictions == captions) & mask
            correct_predictions += correct.sum().item()
            total_tokens += mask.sum().item()

    # Průměrná testovací loss a accuracy
    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct_predictions / total_tokens

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_test_loss, accuracy
