import numpy as np
from PIL import Image
import os
from util import semantic_to_mask
import torch
import cv2
from data_loader import get_dataloader
import torch.nn as nn


@torch.no_grad()
def generate_test():

    input_dir = "J:\\Dataset\\NPC_V1\\val\\image"
    output_dir = "J:\\Dataset\\NPC_V1\\output"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load("./exp/21_RendDANet_0.6905.pth", map_location='cpu').module

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)
    model.eval()
    labels = [0, 1, 2]
    files = os.listdir(input_dir)
    for file in files:
        image = np.load(os.path.join(input_dir, file))
        for i in range(image.shape[2]):
            image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (np.max(image[:, :, i]) - np.min(image[:, :, i]))
        image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(dim=0)
        output = model(image)['fine'].cpu().detach().numpy()
        pred = semantic_to_mask(output, labels).squeeze()
        size = pred.shape[0]
        print(pred.shape)

        # 红色NPC,绿色NPL
        color = np.zeros([size, size, 3], dtype=np.uint8)
        npc = pred == 1
        npl = pred == 2
        color[:, :, 0][npc] = 255
        color[:, :, 1][npl] = 255
        png_slice = Image.fromarray(color)
        png_slice.save(os.path.join(output_dir, file.split('.')[0]) + ".png")


if __name__ == "__main__":
    generate_test()
    exit(0)
