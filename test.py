import numpy as np
from PIL import Image
import os
from util import semantic_to_mask, get_confusion_matrix, get_miou, get_classification_report
import torch
import torch.nn.functional as F
import cv2
from data_loader import get_dataloader
import torch.nn as nn
import pandas as pd
from sklearn.metrics import classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


@torch.no_grad()
def generate_test():
    input_dir = "../data/NPC20_V1/val/image"
    mask_dir = "../data/NPC20_V1/val/mask"
    output_dir = "../data/NPC20_V1/out/"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model_names = sorted(os.listdir("./final_models"))

    for model_name in model_names:

        print(model_name)
        model = torch.load("./final_models/" + model_name, map_location='cpu').module
        print(model)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model = model.to(device)
        model.eval()
        labels = [0, 1, 2]
        files = sorted(os.listdir(input_dir))
        df = pd.DataFrame(
            columns=["ID", 'IoU1', 'IoU2', 'Dice1', 'Dice2', 'Recall1', 'Recall2', 'Precision1', 'Precision2',
                     'N_Tumor', "N_MLN"])
        df['ID'] = list(map(lambda x: x.split("_")[0] + "_" + x.split("_")[1], files))
        recall1 = []
        precision1 = []
        dice1 = []
        iou1 = []

        recall2 = []
        precision2 = []
        dice2 = []
        iou2 = []

        tumor = []
        mln = []
        for file in files:

            image = np.load(os.path.join(input_dir, file))
            for i in range(image.shape[2]):
                image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (
                            np.max(image[:, :, i]) - np.min(image[:, :, i]))
            image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(dim=0).to(device)

            if model_name.split('_')[1] == "NewModel":
                _, _, final = model(image)
                output = final['fine'].cpu().detach().numpy()
            elif model_name.split('_')[1] == "baseBEM":
                _, final = model(image)
                output = final['fine'].cpu().detach().numpy()
            elif model_name.split('_')[1] == "BASNet":
                pred, _, _, _, _, _, _, _ = model(image)
                output = pred.cpu().detach().numpy()
            elif model_name.split('_')[1] == "HRNet":
                pred = model(image)
                output = pred.cpu().detach().numpy()
            elif model_name.split('_')[1] in ["baseSEM", "basePEMSEM"]:
                _, pred = model(image)
                output = pred.cpu().detach().numpy()
            elif model_name.split('_')[1] == "baseSEM":
                _, pred = model(image)
                output = pred.cpu().detach().numpy()
            elif model_name.split('_')[1] == "HRNet":
                out = model(image)
                print(out)
                _, pred = model(image)
                output = pred.cpu().detach().numpy()
            else:
                pred = model(image)
                output = pred.cpu().detach().numpy()

            pred = semantic_to_mask(output, labels).squeeze()
            mask = np.load(os.path.join(mask_dir, file.split('.')[0]) + ".npy")
            cm = get_confusion_matrix(mask, pred, labels)
            IoUs = get_miou(cm)
            cls_report = classification_report(y_true=mask.flatten(), y_pred=pred.flatten(), output_dict=True,
                                               labels=labels)

            iou1.append(IoUs[1])
            dice1.append(cls_report['1']['f1-score'])
            precision1.append(cls_report['1']['precision'])
            recall1.append(cls_report['1']['recall'])
            tumor.append(cls_report['1']['support'])

            iou2.append(IoUs[2])
            dice2.append(cls_report['2']['f1-score'])
            precision2.append(cls_report['2']['precision'])
            recall2.append(cls_report['2']['recall'])
            mln.append(cls_report['2']['support'])

            print("\n*******")
            print(file)
            print(IoUs[1], cls_report['1']['f1-score'], cls_report['1']['precision'], cls_report['1']['recall'], cls_report['1']['support'])
            print(IoUs[2], cls_report['2']['f1-score'], cls_report['2']['precision'], cls_report['2']['recall'], cls_report['2']['support'])
            print("*******\n")

            # 红色NPC,绿色NPL
            # size = pred.shape[0]
            # color = np.zeros([size, size, 3], dtype=np.uint8)
            # npc = pred == 1
            # npl = pred == 2
            # color[:, :, 0][npc] = 255
            # color[:, :, 1][npl] = 255
            # png_slice = Image.fromarray(color)
            # png_slice.save(os.path.join(output_dir + model_name.split('_')[1], file.split('.')[0]) + "_" + str((miou != 0).sum() - 1) + "_" + str(score)[:6] + ".png")

        df['IoU1'] = iou1
        df['Dice1'] = dice1
        df['Recall1'] = recall1
        df['Precision1'] = precision1

        df['IoU2'] = iou2
        df['Dice2'] = dice2
        df['Recall2'] = recall2
        df['Precision2'] = precision2

        df['N_Tumor'] = tumor
        df['N_MLN'] = mln
        df.to_csv("./" + model_name.split("_")[1] + ".csv")


if __name__ == "__main__":
    generate_test()
    exit(0)
