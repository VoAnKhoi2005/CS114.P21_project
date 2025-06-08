import torch

def val_loop(dataloader, model, loss_func, device):
    model.eval()
    test_loss, correct = 0, 0
    total_samples = 0
    total_batches = 0

    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            pred = model(x)
            test_loss += loss_func(pred, y).item() * batch_size  # sum loss over batch
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total_samples += batch_size
            total_batches += 1

    avg_loss = test_loss / total_samples
    accuracy = 100 * correct / total_samples
    print(f"Test_Accuracy: {accuracy:>0.1f}%, Test_Loss: {avg_loss:>8f}")
    return avg_loss, accuracy
