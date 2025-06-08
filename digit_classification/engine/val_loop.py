# src/engine/val_loop.py
import torch

def val_loop(dataloader, model, loss_func, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test_Accuracy: {(100 * correct):>0.1f}%, Test_Loss: {test_loss:>8f}")
    return test_loss, 100 * correct