import time

import torch


def train_loop(dataloader, model, loss_func, optimizer, device, max_batches=None):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    start_time = time.time()

    for batch_idx, (images, labels, _) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        loss = loss_func(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        total_samples += batch_size

        if batch_idx % 500 == 0:
            print(f"Batch {batch_idx}: loss={loss.item():.6f}, samples_processed={total_samples}")

        if max_batches is not None and batch_idx + 1 >= max_batches:
            break

    end_time = time.time()
    elapsed = end_time - start_time

    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = 100 * correct / total_samples if total_samples > 0 else 0
    print(f"Train Accuracy: {accuracy:.1f}%, Train Loss: {avg_loss:.6f}")
    print(f"Training time: {elapsed:.2f} seconds")
    return avg_loss, accuracy
