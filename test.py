import numpy as np
from PIL import Image
import os
from util import semantic_to_mask, get_confusion_matrix, get_miou, get_classification_report
import torch
import cv2
from data_loader import get_dataloader
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES'] = '5'


@torch.no_grad()
def generate_test():

    input_dir = "../data/NPC20_V1/val/image"
    mask_dir = "../data/NPC20_V1/val/mask"
    output_dir = "../data/NPC20_V1/out/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_names = os.listdir("./final_models")

    # model = torch.load("./exp/21_RendDANet_0.6905.pth", map_location='cpu').module

    for model_name in model_names:
        model = torch.load("./final_models/" + model_name, map_location='cpu').module

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
            image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(dim=0).to(device)

            if model_name.split('_')[1] == "NewModel":
                aux_pred, pred, final = model(image)
                output = final['fine'].cpu().detach().numpy()
            elif model_name.split('_')[1] == "BASNet":
                pred, _, _, _, _, _, _, _ = model(image)
                output = pred.cpu().detach().numpy()
            elif model_name.split('_')[1] == "HRNet":
                _, pred = model(image)
                output = pred.cpu().detach().numpy()
            else:
                pred = model(image)
                output = pred.cpu().detach().numpy()

            pred = semantic_to_mask(output, labels).squeeze()
            mask = np.load(os.path.join(mask_dir, file.split('.')[0]) + ".npy")
            cm = get_confusion_matrix(mask, pred, labels)
            miou = get_miou(cm)
            print(miou)
            score = (miou[1] + miou[2]) / ((miou != 0).sum() - 1 + 1e-6)
            size = pred.shape[0]
            print(model_name.split('_')[1], file)

            # 红色NPC,绿色NPL
            color = np.zeros([size, size, 3], dtype=np.uint8)
            npc = pred == 1
            npl = pred == 2
            color[:, :, 0][npc] = 255
            color[:, :, 1][npl] = 255
            png_slice = Image.fromarray(color)
            png_slice.save(os.path.join(output_dir + model_name.split('_')[1], file.split('.')[0]) + "_" + str((miou != 0).sum() - 1) + "_" + str(score)[:6] + ".png")


if __name__ == "__main__":
    generate_test()
    exit(0)
