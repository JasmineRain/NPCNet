import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from data_transform import HorizontalFlip, VerticalFlip, Rotate, ToTensor, Resize


class NPCDataset(Dataset):

    def __init__(self, img_dir, mask_dir, mode='test', smooth=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.smooth = smooth
        self.labels = [0, 1, 2]
        self.images = list(sorted(os.listdir(img_dir)))
        self.masks = list(sorted(os.listdir(mask_dir)))
        self.hf = HorizontalFlip(p=1)
        self.vf = VerticalFlip(p=1)
        self.rt = Rotate(degrees=(90, 180, 270))
        self.rs = Resize(scales=[(320, 320), (192, 192), (384, 384), (128, 128)], p=0.5)
        self.tt = ToTensor()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        image = np.load(os.path.join(self.img_dir, self.images[item]))
        mask = np.load(os.path.join(self.mask_dir, self.masks[item]))

        if self.mode == "train":
            seed = np.random.randint(0, 4, 1)
            if seed == 0:
                pass
            elif seed == 1:
                image, mask = self.hf(image, mask)
            elif seed == 2:
                image, mask = self.vf(image, mask)
            elif seed == 3:
                image, mask = self.rt(image, mask)

            image, mask = self.tt(image, mask, labels=self.labels)

        elif self.mode == 'val':

            image, mask = self.tt(image, mask, labels=self.labels)

        elif self.mode == 'test':

            return image, mask, self.images[item]

        else:
            print("invalid transform mode")

        return image, mask


def get_dataloader(img_dir, mask_dir, batch_size, num_workers, mode="train", smooth=False):

    if mode == "train":
        train_dataset = NPCDataset(img_dir, mask_dir, mode="train", smooth=False)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        return train_dataloader
    elif mode == "test":
        test_dataset = NPCDataset(img_dir, mask_dir, mode='test', smooth=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_dataloader
    else:
        val_dataset = NPCDataset(img_dir, mask_dir, mode='val', smooth=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return val_dataloader


if __name__ == "__main__":

    train_image_dir = "../data/Multi_V1/train/image"
    train_label_dir = "../data/Multi_V1/train/label"
    val_image_dir = "../data/Multi_V1/val/image"
    val_label_dir = "../data/Multi_V1/val/label"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(train_image_dir, train_label_dir, batch_size=1, num_workers=1, mode="train")
    val_loader = get_dataloader(val_image_dir, val_label_dir, batch_size=1, num_workers=1, mode="val")

    train_dataset = NPCDataset(train_image_dir, train_label_dir, mode="train")
    image, mask = train_dataset[0]
    print(image.shape)
    print(mask.shape)

    # for image, mask in val_loader:
    #
    #     image = image.to(device)
    #     image = (image > 0.5).float()
    #     mask = mask.to(device)
    #     print(image.shape)
    #     print(mask.shape)
    #     print(torch.unique(mask))
    #     print(torch.unique(image))
    #
    #     # plt.imshow(image[0, 0, :, :])
    #     # plt.pause(0.1)
    #     # plt.imshow(mask[0, 0, :, :])
    #     # plt.pause(0.1)
    #
    #     break
