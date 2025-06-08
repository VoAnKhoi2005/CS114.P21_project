# src/engine/train_loop.py
import torch

def train_loop(dataloader, model, loss_func, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    total_loss, correct = 0, 0
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        loss = loss_func(pred, labels)

        total_loss += loss.item()
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current = batch * len(images)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / size
    print(f"Train_Accuracy: {accuracy:>0.1f}%, Train_Loss: {avg_loss:>8f}")
    return avg_loss, accuracy