from matplotlib import pyplot as plt


class History:
    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []

def plot_history(history):
    epochs = range(1, len(history.train_loss) + 1)

    plt.figure(figsize=(12, 5))

    # Plot train
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.train_loss, 'bo-', label='Train Loss')
    plt.plot(epochs, history.test_loss, 'ro-', label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)

    # Plot test
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.train_accuracy, 'bo-', label='Train Accuracy')
    plt.plot(epochs, history.test_accuracy, 'ro-', label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)

    plt.tight_layout()
    plt.show()