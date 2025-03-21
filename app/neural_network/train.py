import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm


def train_model(model, train_loader, val_loader, learning_rate=3e-4, epochs=100, device="cuda" if torch.cuda.is_available() else "cpu", checkpoint_path="checkpoints"):
    """
    Trénink modelu pro převod matematických výrazů do LaTeX.
    """
    # Přesun modelu na zvolené zařízení (GPU/CPU)
    model = model.to(device)

    # Definice loss funkce a optimizeru
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding tokens
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.98))

    # Learning rate scheduler - lineární warm-up a poté decay
    def lr_lambda(current_step, warmup_steps=5, decay_rate=0.95):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return decay_rate ** (current_step - warmup_steps)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Early stopping
    patience = 7
    early_stopping_counter = 0

    # Nejlepší validační loss
    best_val_loss = float("inf")
    best_accuracy = 0.0

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Trénink
        model.train()
        train_loss = 0.0
        total_correct = 0
        total_tokens = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            # Optimalizace
            optimizer.step()

            # Akumulace loss
            train_loss += loss.item()

            # Výpočet přesnosti
            _, predictions = outputs.max(1)
            mask = targets != 0  # Ignorování padding tokenů
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            # Log každých 100 batch
            if (i + 1) % 100 == 0:
                batch_accuracy = correct.sum().item() / max(1, mask.sum().item())
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}")

        # Průměrná tréninková loss a accuracy
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = total_correct / max(1, total_tokens)

        # Validace
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tokens = 0

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

                # Výpočet přesnosti
                _, predictions = outputs.max(1)
                mask = targets != 0  # Ignorování padding tokenů
                correct = (predictions == targets) & mask
                val_correct += correct.sum().item()
                val_tokens += mask.sum().item()

        # Průměrná validační loss a accuracy
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / max(1, val_tokens)

        # Learning rate scheduler
        scheduler.step()

        # Výpis statistik
        time_elapsed = time.time() - start_time
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print(f"Time: {time_elapsed:.2f}s")

        # Uložení nejlepšího modelu na základě přesnosti
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                    "accuracy": val_accuracy,
                },
                f"{checkpoint_path}/best_model_by_accuracy.pth",
            )
            print(f"Model saved with best accuracy: {val_accuracy:.4f}!")
            # Reset early stopping counter
            early_stopping_counter = 0
        # Uložení nejlepšího modelu podle loss
        elif avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_val_loss,
                    "accuracy": val_accuracy,
                },
                f"{checkpoint_path}/best_model.pth",
            )
            print(f"Model saved with best loss: {best_val_loss:.4f}!")
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Validation metrics did not improve. Early stopping counter: {early_stopping_counter}/{patience}")

            if early_stopping_counter >= patience:
                print("Early stopping triggered! Ending training.")
                break

        # Uložení checkpointu každých 5 epoch
        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_val_loss,
                    "accuracy": val_accuracy,
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
