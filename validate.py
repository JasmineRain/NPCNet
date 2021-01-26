import numpy as np
from PIL import Image
import os
from util import semantic_to_mask, get_confusion_matrix, get_miou, get_classification_report
import torch
import pydicom
import torch.nn as nn
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def get_boundary(image):
    points = np.zeros(shape=image.shape)
    for x in range(0, len(image)):
        for y in range(0, len(image[x])):
            if image[x][y] != 2:
                continue
            # strategy 1
            # if 0 in image[y-1:y+2, x-1:x+2]:
            #     points.append((y, x))

            # strategy 2
            if x == 0 or y == 0 or y == len(image) - 1 or x == len(image[y]) - 1:
                points[x][y] = 1
                # points.append((y, x))
                continue
            if 0 in [image[x][y - 1], image[x][y + 1], image[x - 1][y], image[x + 1][y]]:
                points[x][y] = 1
    return points


@torch.no_grad()
def validate():
    input_dir = "../data/NPC20_V1/test_image"
    mask_dir = "../data/NPC20_V1/test_label"
    dicom_dir = "../data/NPC20_V1/raw_dicom"
    output_dir = "../data/NPC20_V1/out_dicom_lympha"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load("./test_model.pth", map_location='cpu').module

    model = model.to(device)
    model.eval()

    labels = [0, 1, 2]

    files = os.listdir(input_dir)
    cm = np.zeros([3, 3])
    for file in files:
        image = np.load(os.path.join(input_dir, file))
        for i in range(image.shape[2]):
            image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (
                    np.max(image[:, :, i]) - np.min(image[:, :, i]))
        image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(dim=0).to(device)

        t1_files = os.listdir(os.path.join(dicom_dir, file.split('.')[0].split('_')[0], "T1"))
        t2_files = os.listdir(os.path.join(dicom_dir, file.split('.')[0].split('_')[0], "T2"))
        t1c_files = os.listdir(os.path.join(dicom_dir, file.split('.')[0].split('_')[0], "T1C"))
        t1_files.sort(key=lambda x: x.split('.')[0][-2:], reverse=True)
        t2_files.sort(key=lambda x: x.split('.')[0][-2:], reverse=True)
        t1c_files.sort(key=lambda x: x.split('.')[0][-2:], reverse=True)

        idx = int(file.split('.')[0].split('_')[1])
        patient_ID = file.split('_')[0]

        t1 = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T1", t1_files[idx]), force=True)
        t1.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        t2 = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T2", t2_files[idx]), force=True)
        t2.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        t1c = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T1C", t1c_files[idx]), force=True)
        t1c.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        t1_arr = t1.pixel_array
        t2_arr = t2.pixel_array
        t1c_arr = t1c.pixel_array

        t1 = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T1", t1_files[idx]))
        t2 = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T2", t2_files[idx]))
        t1c = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T1C", t1c_files[idx]))
        mask = np.load(os.path.join(mask_dir, file.split('.')[0]) + ".npy")

        _, _, final = model(image)
        output = final['fine'].cpu().detach()

        t1_mask = F.interpolate(output, mode='bilinear', size=t1_arr.shape, align_corners=True).numpy()
        t2_mask = F.interpolate(output, mode='bilinear', size=t2_arr.shape, align_corners=True).numpy()
        t1c_mask = F.interpolate(output, mode='bilinear', size=t1c_arr.shape, align_corners=True).numpy()

        t1_gt = F.interpolate(torch.from_numpy(mask).unsqueeze(dim=0).unsqueeze(dim=0), mode='nearest', size=t1_arr.shape).squeeze().numpy()
        t2_gt = F.interpolate(torch.from_numpy(mask).unsqueeze(dim=0).unsqueeze(dim=0), mode='nearest', size=t2_arr.shape).squeeze().numpy()
        t1c_gt = F.interpolate(torch.from_numpy(mask).unsqueeze(dim=0).unsqueeze(dim=0), mode='nearest', size=t1c_arr.shape).squeeze().numpy()

        output = output.numpy()

        pred = semantic_to_mask(output, labels).squeeze()
        t1_mask = semantic_to_mask(t1_mask, labels).squeeze()
        t2_mask = semantic_to_mask(t2_mask, labels).squeeze()
        t1c_mask = semantic_to_mask(t1c_mask, labels).squeeze()

        t1_boundary = get_boundary(t1_mask)
        t2_boundary = get_boundary(t2_mask)
        t1c_boundary = get_boundary(t1c_mask)

        t1_gt_boundary = get_boundary(t1_gt)
        t2_gt_boundary = get_boundary(t2_gt)
        t1c_gt_boundary = get_boundary(t1c_gt)

        t1_arr[t1_boundary == 1] = 2000
        t2_arr[t2_boundary == 1] = 2000
        t1c_arr[t1c_boundary == 1] = 2000

        t1_arr[t1_gt_boundary == 1] = 3000
        t2_arr[t2_gt_boundary == 1] = 3000
        t1c_arr[t1c_gt_boundary == 1] = 3000

        t1.PixelData = t1_arr.tobytes()
        t2.PixelData = t2_arr.tobytes()
        t1c.PixelData = t1c_arr.tobytes()

        # write file
        cm_slice = get_confusion_matrix(mask, pred, labels)
        cm += cm_slice
        miou_slice = get_miou(cm_slice)
        print(file, miou_slice)
        score_slice = 2 * miou_slice[1] / (1 + miou_slice[1] + 1e-6)

        # Tumor
        t1.save_as(os.path.join(output_dir, patient_ID, "T1", t1_files[idx].split('.')[0] + "_" + str(score_slice) + ".dcm"))
        t2.save_as(os.path.join(output_dir, patient_ID, "T2", t1_files[idx].split('.')[0] + "_" + str(score_slice) + ".dcm"))
        t1c.save_as(os.path.join(output_dir, patient_ID, "T1C", t1_files[idx].split('.')[0] + "_" + str(score_slice) + ".dcm"))

        # MLN
        # t1.save_as(os.path.join(output_dir, patient_ID, "T1", t1_files[idx].split('.')[0] + ".dcm"))
        # t2.save_as(os.path.join(output_dir, patient_ID, "T2", t1_files[idx].split('.')[0] + ".dcm"))
        # t1c.save_as(os.path.join(output_dir, patient_ID, "T1C", t1_files[idx].split('.')[0] + ".dcm"))

    final_miou = get_miou(cm)
    final_score = 2 * final_miou[1] / (1 + final_miou[1] + 1e-6)
    print("----- final score -----")
    print(final_miou, final_score)


@torch.no_grad()
def validate_fix():
    input_dir = "../data/NPC20_V1/test_image"
    mask_dir = "../data/NPC20_V1/test_label"
    dicom_dir = "../data/NPC20_V1/raw_dicom"
    output_dir = "../data/NPC20_V1/out_dicom_lympha"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load("./test_model.pth", map_location='cpu').module

    model = model.to(device)
    model.eval()

    labels = [0, 1, 2]

    files = os.listdir(input_dir)
    cm = np.zeros([3, 3])
    for file in files:
        image = np.load(os.path.join(input_dir, file))
        for i in range(image.shape[2]):
            image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (
                    np.max(image[:, :, i]) - np.min(image[:, :, i]))
        image = torch.from_numpy(image.transpose((2, 0, 1))).unsqueeze(dim=0).to(device)

        # 输入和标签均转为512
        if image.shape[-1] != 512:
            image = F.interpolate(image, mode='bilinear', align_corners=True, size=(512, 512))

        mask = np.load(os.path.join(mask_dir, file.split('.')[0]) + ".npy")
        mask = F.interpolate(torch.from_numpy(mask.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0),
                                    mode='bilinear', size=(512, 512), align_corners=True)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = mask.squeeze().numpy().astype(np.uint8)

        t1_files = os.listdir(os.path.join(dicom_dir, file.split('.')[0].split('_')[0], "T1"))
        t2_files = os.listdir(os.path.join(dicom_dir, file.split('.')[0].split('_')[0], "T2"))
        t1c_files = os.listdir(os.path.join(dicom_dir, file.split('.')[0].split('_')[0], "T1C"))
        t1_files.sort(key=lambda x: x.split('.')[0][-2:], reverse=True)
        t2_files.sort(key=lambda x: x.split('.')[0][-2:], reverse=True)
        t1c_files.sort(key=lambda x: x.split('.')[0][-2:], reverse=True)

        idx = int(file.split('.')[0].split('_')[1])
        patient_ID = file.split('_')[0]

        t1 = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T1", t1_files[idx]), force=True)
        t1.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        t2 = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T2", t2_files[idx]), force=True)
        t2.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        t1c = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T1C", t1c_files[idx]), force=True)
        t1c.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        t1_arr = t1.pixel_array
        t2_arr = t2.pixel_array
        t1c_arr = t1c.pixel_array

        if not t1_arr.shape[0] != 512:
            t1_arr = F.interpolate(torch.from_numpy(t1_arr.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0),
                                 mode='bilinear', size=(512, 512), align_corners=True).squeeze().numpy().astype(">i2")
        if not t2_arr.shape[0] != 512:
            t2_arr = F.interpolate(torch.from_numpy(t2_arr.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0),
                                 mode='bilinear', size=(512, 512), align_corners=True).squeeze().numpy().astype(">i2")
        if not t1c_arr.shape[0] != 512:
            t1c_arr = F.interpolate(torch.from_numpy(t1c_arr.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0),
                                 mode='bilinear', size=(512, 512), align_corners=True).squeeze().numpy().astype(">i2")

        t1 = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T1", t1_files[idx]))
        t2 = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T2", t2_files[idx]))
        t1c = pydicom.dcmread(os.path.join(dicom_dir, patient_ID, "T1C", t1c_files[idx]))

        _, _, final = model(image)
        output = final['fine'].cpu().detach()

        output = output.numpy()

        pred = semantic_to_mask(output, labels).squeeze()

        pred_boundary = get_boundary(pred)
        gt_boundary = get_boundary(mask)

        t1_arr[pred_boundary == 1] = 2000
        t2_arr[pred_boundary == 1] = 2000
        t1c_arr[pred_boundary == 1] = 2000

        t1_arr[gt_boundary == 1] = 3000
        t2_arr[gt_boundary == 1] = 3000
        t1c_arr[gt_boundary == 1] = 3000

        t1.PixelData = t1_arr.tobytes()
        t2.PixelData = t2_arr.tobytes()
        t1c.PixelData = t1c_arr.tobytes()

        # write file
        cm_slice = get_confusion_matrix(mask, pred, labels)
        cm += cm_slice
        miou_slice = get_miou(cm_slice)
        print(file, miou_slice)
        score_slice = 2 * miou_slice[1] / (1 + miou_slice[1] + 1e-6)

        # Tumor
        t1.save_as(os.path.join(output_dir, patient_ID, "T1", t1_files[idx].split('.')[0] + "_" + str(score_slice) + ".dcm"))
        t2.save_as(os.path.join(output_dir, patient_ID, "T2", t1_files[idx].split('.')[0] + "_" + str(score_slice) + ".dcm"))
        t1c.save_as(os.path.join(output_dir, patient_ID, "T1C", t1_files[idx].split('.')[0] + "_" + str(score_slice) + ".dcm"))

        # MLN
        # t1.save_as(os.path.join(output_dir, patient_ID, "T1", t1_files[idx].split('.')[0] + ".dcm"))
        # t2.save_as(os.path.join(output_dir, patient_ID, "T2", t1_files[idx].split('.')[0] + ".dcm"))
        # t1c.save_as(os.path.join(output_dir, patient_ID, "T1C", t1_files[idx].split('.')[0] + ".dcm"))

    final_miou = get_miou(cm)
    final_score = 2 * final_miou[1] / (1 + final_miou[1] + 1e-6)
    print("----- final score -----")
    print(final_miou, final_score)


if __name__ == "__main__":
    validate()
    exit(0)
