import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
from torch.nn import CTCLoss
from torch.amp import autocast, GradScaler
# from torch.utils.data import Subset

def coverage_loss(attention_weights, coverage_vector):
    """Calculate coverage loss to prevent under/over-attention."""
    min_value = torch.min(attention_weights, coverage_vector)
    loss = torch.sum(min_value, dim=2)
    return torch.mean(loss)

def train_model(model, train_loader, val_loader, learning_rate=3e-4, epochs=100,
                device="cuda" if torch.cuda.is_available() else "cpu", 
                checkpoint_path="checkpoints"):
    """
    Trénink modelu pro převod matematických výrazů do LaTeX.
    Používá kombinaci CTC a CrossEntropy loss s mixed precision training.
    """
    # Přesun modelu na zvolené zařízení (GPU/CPU)
    model = model.to(device)

    # Definice loss funkcí a optimizeru
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    ce_criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                        weight_decay=1e-4, betas=(0.9, 0.98))   # Weight decay changed from 0.01 to 1e-4

    # Initialize the scheduler and scaler for mixed precision training
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=epochs * len(train_loader),
        pct_start=0.30,
        div_factor=10,
        final_div_factor=500,
        anneal_strategy='cos'
    )
    scaler = GradScaler(device)

    # Loss weights
    ctc_weight = 0.5
    ce_weight = 0.5

    # Early stopping parameters
    patience = 10
    early_stopping_counter = 0
    best_val_loss = float("inf")
    best_accuracy = 0.0
    global_step = 0
    accumulation_steps = 2

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Starting Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Training
        model.train()
        train_loss = 0.0
        total_correct = 0
        total_tokens = 0
        start_time = time.time()

        for i, (images, captions, lengths) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            captions = captions.to(device)
            
            # Mixed precision training
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                # Forward pass
                outputs = model(images, captions, lengths)
                batch_size = outputs.size(0)

                # Prepare targets for CTC loss
                targets = captions.clone()
                target_lengths = torch.as_tensor(lengths, device=device)
                input_lengths = torch.full((batch_size,), outputs.size(1), device=device)

                # Calculate both losses
                outputs_flat = outputs.view(-1, outputs.size(2))
                targets_flat = captions.view(-1)
                
                # CTC loss
                log_probs = outputs.log_softmax(2)
                ctc_loss = ctc_criterion(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
                
                # Cross entropy loss
                ce_loss = ce_criterion(outputs_flat, targets_flat)
                
                # Combine losses
                loss = ctc_weight * ctc_loss + ce_weight * ce_loss

            # Backward pass with gradient scaling
            scaled_loss = scaler.scale(loss) / accumulation_steps
            scaled_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            train_loss += loss.item()

            # Calculate accuracy
            _, predictions = outputs_flat.max(1)
            mask = targets_flat != 0
            correct = (predictions == targets_flat) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

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

                loss = ce_criterion(outputs, targets)
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

# New version: Token-Level Accuracy with F1 Score
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
    total_sequences = 0
    correct_sequences = 0

    with torch.no_grad():
        for images, captions, lengths in tqdm(test_loader):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions, lengths)
            batch_size = outputs.size(0)
            outputs_flat = outputs.view(-1, outputs.size(2))
            targets_flat = captions.view(-1)

            # Calculate loss
            loss = criterion(outputs_flat, targets_flat)
            test_loss += loss.item()

            # Get predictions
            _, predictions = outputs.max(2)

            # Mask padding tokens
            mask = captions != 0

            # Token-level accuracy
            correct = (predictions == captions) & mask
            correct_predictions += correct.sum().item()
            total_tokens += mask.sum().item()

            # Sequence-level accuracy
            for i in range(batch_size):
                target_seq = captions[i, :lengths[i]].tolist()  # Ground truth sequence
                pred_seq = predictions[i, :lengths[i]].tolist()  # Predicted sequence
                if target_seq == pred_seq:
                    correct_sequences += 1
                total_sequences += 1

    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    token_accuracy = correct_predictions / total_tokens
    sequence_accuracy = correct_sequences / total_sequences

    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Token-Level Accuracy: {token_accuracy:.4f}")
    print(f"Sequence-Level Accuracy: {sequence_accuracy:.4f}")

    return avg_test_loss, token_accuracy, sequence_accuracy