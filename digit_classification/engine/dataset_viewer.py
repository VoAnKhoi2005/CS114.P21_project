import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def show_batch(test_loader, img_size=224):
    cols, rows = 3, 3
    batch_iter = iter(test_loader)

    fig, axs = plt.subplots(rows, cols, figsize=(img_size, img_size))
    plt.subplots_adjust(bottom=0.2)  # Make space for button

    def plot_next(_event=None):
        nonlocal batch_iter
        try:
            batch_imgs, batch_labels, batch_paths = next(batch_iter)
        except StopIteration:
            print("No more batches.")
            return

        for ax in axs.flat:
            ax.clear()
            ax.axis('off')

        for i in range(min(cols * rows, len(batch_imgs))):
            img = batch_imgs[i]
            path = batch_paths[i]
            ax = axs.flat[i]

            title = os.path.basename(path)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

            if img.shape[0] == 1:
                ax.imshow(img[0].cpu(), cmap='gray')
            else:
                ax.imshow(img.permute(1, 2, 0).cpu())

        fig.canvas.draw_idle()

    # Add button
    ax_next = plt.axes([0.4, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
    btn_next = Button(ax_next, 'Next Batch')
    btn_next.on_clicked(plot_next)

    plot_next()  # Show the first batch
    plt.show()
