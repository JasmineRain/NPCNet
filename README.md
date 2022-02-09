# NPCNet
The official implementation of work entitled "NPCNet: Jointly Segment Primary Nasopharyngeal Carcinoma Tumor and Metastatic Lymph Nodes in MR Images", accepted by IEEE Transactions on Medical Imaging, 2022. The paper is now under publishing and now available at [IEEE Early Access](https://ieeexplore.ieee.org/document/9684475)

# Method
![Model Architecture](https://github.com/JasmineRain/NPCNet/doc/arc.png)

# Training

## Requirements

- Python>=3.7
- PyTorch ≥ 1.4 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- Other libs in the requirements.txt file

## Prepare datasets

You should prepare your own dataset according to the dataloader.py file. The input MRI images are concatenated in channel (sequence is T1, T2, T1c) and saved in .npy format with data type of float32. The GT files (segmentation mask) should be saved in .npy format with data type of uint8.

## Model weights

The weights trained on our dataset are available at [Google Cloud](https://drive.google.com/drive/folders/13C8NXUF4jP7Ix2-1Mgya0mpsH2A6b5Me?usp=sharing).

## Training details

To achieve the best results and avoid unstable training, you can train the model according to these steps:
1. Train a model without the proposed BEM (which is backbone + PEM + SEM) and save the weights.
2. Modify the model definition code by adding the proposed BEM (which is NPCNet).
3. Train the whole NPCNet with the pretrained weights in step 1 (parameters in the BEM are initialized randomly). The learning rate for backbone, PEM, and SEM is 1e-3 and the learning rate for BEM is 1e-2.

# Q&A
Leave your question in the issue of this repo. We would answer your question ASAP.

# Citation
```
@ARTICLE{9684475,
  author={Li, Yang and Dan, Tingting and Li, Haojiang and Chen, Jiazhou and Peng, Hong and Liu, Lizhi and Cai, Hongmin},
  journal={IEEE Transactions on Medical Imaging},
  title={NPCNet: Jointly Segment Primary Nasopharyngeal Carcinoma Tumors and Metastatic Lymph Nodes in MR Images},
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2022.3144274}
}
```
