import numpy as np
from PIL import Image
import os
from util import semantic_to_mask, get_confusion_matrix, get_miou, get_classification_report
import torch
import cv2
from data_loader import get_dataloader
import torch.nn as nn


def generate_GT():
    gt_src = "../data/NPC20_V1/val/mask"
    output_dir = "../data/NPC20_V1/out/GT/"
    files = os.listdir(gt_src)
    for file in files:
        mask = np.load(os.path.join(gt_src, file)).astype(np.uint8)
        size = mask.shape[0]
        color = np.zeros([size, size, 3], dtype=np.uint8)
        npc = mask == 1
        npl = mask == 2
        color[:, :, 0][npc] = 255
        color[:, :, 1][npl] = 255

        png = Image.fromarray(color)
        png.save(output_dir + file.split('.')[0] + ".png")


def vis_t1c():
    t1c_src = "../data/NPC20_V1/val/image"
    output_dir = "../data/NPC20_V1/out/Raw_T2/"
    files = os.listdir(t1c_src)
    for file in files:
        mask = np.load(os.path.join(t1c_src, file))[:, :, 1]
        mask = mask / mask.max()
        mask = (mask * 255).astype(np.uint8)
        png = Image.fromarray(mask)
        png.save(output_dir + file.split('.')[0] + ".png")


def crop_mask():
    temp = "./visualization/out_add/RANet/"
    output = "./visualization/Mask_Center_add/RANet/"

    files = os.listdir(temp)

    for file in files:
        pred = Image.open(temp + file)

        if file.split('_')[1] == "10071797":
            pred = pred.crop((180, 125, 430, 325))
        elif file.split('_')[1] == "10104173":
            pred = pred.crop((200, 150, 350, 300))
        elif file.split('_')[1] == "10097057":
            pred = pred.crop((175, 175, 325, 325))
        else:
            pred = pred.crop((90, 170, 380, 360))
            pred.save(output + file)
            continue

        pred.save(output + file.split('_')[0] + "_" + file.split('_')[1] + "_" + file.split('_')[2] + ".png")


if __name__ == "__main__":
    crop_mask()