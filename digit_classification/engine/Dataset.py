import torch
from PIL import UnidentifiedImageError
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.io import decode_image


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.img_paths = images
        self.img_labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        try:
            image = Image.open(self.img_paths[index]).convert("L")
        except UnidentifiedImageError:
            # print(f"Warning: Skipping corrupted image {self.img_paths[index]}")
            image = Image.new("L", (128, 128))
        label = self.img_labels[index]

        if self.transform:
            image = self.transform(image)

        import torch
        return image, torch.tensor(label, dtype=torch.long)


class CustomIterableDataset(IterableDataset):
    def __init__(self, image_lists, labels, transform=None):
        self.image_lists = image_lists
        self.labels = labels
        self.transform = transform

    def __read_image(self, idx):
        img_path = self.image_lists[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            label = torch.tensor(self.labels[idx])
            if self.transform:
                image = self.transform(image)
            return image, label, img_path
        except Exception as e:
            #print(f"Error reading image: {e}")
            return None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process
            iter_start = 0
            iter_end = len(self.image_lists)
        else:
            per_worker = int(len(self.image_lists) / float(worker_info.num_workers))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.image_lists))

        indices = range(iter_start, iter_end)
        return iter(
            filter(
                lambda x: x is not None,
                map(self.__read_image, indices)
            )
        )
