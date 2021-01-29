import os
import numpy as np


def average():
    dicom_dir = "J:\Dataset\\new\out_dicom"

    dicom_ids = np.array(os.listdir(dicom_dir))
    count = 0
    dice = 0.0
    for dicom_id in dicom_ids:
        images = os.listdir(os.path.join(dicom_dir, dicom_id, "T1C"))
        for image in images:
            score = image.split("_")[1][:-4]
            # print(image)
            if float(score) > 0:
                dice += float(score)
                count += 1
    print(dice, count)
    ave_dice = dice / count
    print(ave_dice)
    return ave_dice


if __name__ == "__main__":
    ans = average()
    exit(0)
