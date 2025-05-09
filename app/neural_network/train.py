import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm


def train_model(model, train_loader, val_loader, learning_rate=3e-4, epochs=100,
                device="cuda" if torch.cuda.is_available() else "cpu", 
                checkpoint_path="checkpoints"):
    """
    Trénink modelu pro převod matematických výrazů do LaTeX.
    Používá batch-wise learning rate scheduling s delším warm-up.
    """
    # Přesun modelu na zvolené zařízení (GPU/CPU)
    model = model.to(device)

    # Definice loss funkce a optimizeru
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # ignore padding tokens
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                        weight_decay=1e-4, betas=(0.9, 0.98))

    # Initialize the scheduler differently
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=epochs * len(train_loader),
        pct_start=0.15, # First 15% for warmup
        div_factor=10,  # Initial lr = max_lr/10
        final_div_factor=1000,  # Final lr = max_lr/1000
        anneal_strategy='cos'
    )

    # Early stopping
    patience = 10
    early_stopping_counter = 0

    # Nejlepší validační loss
    best_val_loss = float("inf")
    best_accuracy = 0.0

    global_step = 0  # track steps for batch-wise scheduling
    accumulation_steps = 2  # Effective batch size = 32*2 = 64

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Starting Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

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
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            # Optimalizace
            scheduler.step()  # update learning rate every batch
            global_step += 1

            # Akumulace loss
            train_loss += loss.item()

            # Výpočet přesnosti
            _, predictions = outputs.max(1)
            mask = targets != 0  # Ignorování padding tokenů
            correct = (predictions == targets) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

            # Log každých 100 batchů
            if (i + 1) % 100 == 0:
                batch_accuracy = correct.sum().item() / max(1, mask.sum().item())
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.4f}, LR: {current_lr:.6f}")

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
                images = images.to(device)
                captions = captions.to(device)

                outputs = model(images, captions, lengths)
                outputs = outputs.view(-1, outputs.size(2))
                targets = captions.view(-1)

                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predictions = outputs.max(1)
                mask = targets != 0
                correct = (predictions == targets) & mask
                val_correct += correct.sum().item()
                val_tokens += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / max(1, val_tokens)

        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"Epoch {epoch+1}: Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"Epoch {epoch+1}: Time: {time_elapsed:.2f}s")
        print(f"Epoch {epoch+1}: Ending Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

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
            early_stopping_counter = 0
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
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    test_loss = 0.0
    correct_predictions = 0
    total_tokens = 0

    with torch.no_grad():
        for images, captions, lengths in tqdm(test_loader):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions, lengths)
            batch_size = outputs.size(0)
            outputs_flat = outputs.view(-1, outputs.size(2))
            targets_flat = captions.view(-1)

            loss = criterion(outputs_flat, targets_flat)
            test_loss += loss.item()

            _, predictions = outputs.max(2)
            mask = captions != 0
            correct = (predictions == captions) & mask
            correct_predictions += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct_predictions / total_tokens

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}")

    return avg_test_loss, accuracy