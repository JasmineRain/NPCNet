import argparse
import numpy as np
import os
from util import semantic_to_mask, mask_to_semantic, get_confusion_matrix, get_miou, get_classification_report
import torch.nn.functional as F
from models.RendPoint import sampling_features
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, adamw
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models import UNetPP, UNet, rf101, DANet, SEDANet, scSEUNet, RendDANet, NewModel, NewModel2, NewModel3, NewModel4, NewModel5,  NewModel6, NewModel7, NewModel8, NewModel9, NewModel10
from loss import LabelSmoothSoftmaxCE, LabelSmoothCE, RendLoss
from utils_Deeplab import SyncBN2d
from models.DeepLabV3_plus import deeplabv3_plus
from models.HRNetOCR import seg_hrnet_ocr
from data_loader import get_dataloader


ALIGN_CORNERS = False


def train_val(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(img_dir=config.train_img_dir, mask_dir=config.train_mask_dir, mode="train",
                                  batch_size=config.batch_size, num_workers=config.num_workers, smooth=config.smooth)
    val_loader = get_dataloader(img_dir=config.val_img_dir, mask_dir=config.val_mask_dir, mode="val",
                                batch_size=config.batch_size, num_workers=config.num_workers)

    writer = SummaryWriter(
        comment="LR_%f_BS_%d_MODEL_%s_DATA_%s" % (config.lr, config.batch_size, config.model_type, config.data_type))

    if config.model_type == "UNet":
        model = UNet()
    elif config.model_type == "UNet++":
        model = UNetPP()
    elif config.model_type == "SEDANet":
        model = SEDANet()
    elif config.model_type == "RendDANet":
        model = RendDANet(nclass=3, backbone="resnet101", norm_layer=nn.BatchNorm2d)
    elif config.model_type == "RefineNet":
        model = rf101()
    elif config.model_type == "DANet":
        model = DANet(backbone='resnet101', nclass=config.output_ch, pretrained=True, norm_layer=nn.BatchNorm2d)
    elif config.model_type == "Deeplabv3+":
        model = deeplabv3_plus.DeepLabv3_plus(in_channels=3, num_classes=8, backend='resnet101', os=16, pretrained=True, norm_layer=nn.BatchNorm2d)
    elif config.model_type == "HRNet_OCR":
        model = seg_hrnet_ocr.get_seg_model()
    elif config.model_type == "NewModel":
        model = NewModel10(nclass=3, backbone="resnet101", norm_layer=nn.BatchNorm2d, pretrained=True)
        src = "./exp/state_dict_0.7027_no_rend.pth"
        pretrained_dict = torch.load(src, map_location='cpu')
        print("load pretrained params from : " + src)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model = UNet()

    if config.iscontinue:
        model = torch.load("./exp/2_NewModel_0.7042.pth").module

    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability

    labels = [0, 1, 2]
    objects = ['背景', '原发灶', '淋巴结']

    for param in model.backbone.parameters():
        param.requires_grad = not config.freeze
    for param in model.da_head.parameters():
        param.requires_grad = not config.freeze
    for param in model.aspp.parameters():
        param.requires_grad = not config.freeze
    for param in model.conv3x3_ocr.parameters():
        param.requires_grad = not config.freeze
    for param in model.ocr_gather_head.parameters():
        param.requires_grad = not config.freeze
    for param in model.ocr_distri_head.parameters():
        param.requires_grad = not config.freeze
    for param in model.aux_head.parameters():
        param.requires_grad = not config.freeze
    for param in model.cls_head.parameters():
        param.requires_grad = not config.freeze


    if config.optimizer == "sgd":
        # optimizer = SGD(model.parameters(), lr=config.lr, weight_decay=1e-4, momentum=0.9)
        optimizer1 = torch.optim.SGD([
            {'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'lr': 1e-3},
            {'params': filter(lambda p: p.requires_grad, model.da_head.parameters()), 'lr': 1e-3},
            {'params': filter(lambda p: p.requires_grad, model.aspp.parameters()), 'lr': 1e-3},
            {'params': filter(lambda p: p.requires_grad, model.conv3x3_ocr.parameters()), 'lr': 1e-3},
            {'params': filter(lambda p: p.requires_grad, model.ocr_gather_head.parameters()), 'lr': 1e-3},
            {'params': filter(lambda p: p.requires_grad, model.ocr_distri_head.parameters()), 'lr': 1e-3},
            {'params': filter(lambda p: p.requires_grad, model.aux_head.parameters()), 'lr': 1e-3},
            {'params': filter(lambda p: p.requires_grad, model.cls_head.parameters()), 'lr': 1e-3},
        ],
            lr=config.lr, momentum=0.9, weight_decay=1e-4
        )
        optimizer2 = torch.optim.SGD(model.rend_head.parameters(), lr=config.lr, momentum=0.9, weight_decay=1e-4)
    elif config.optimizer == "adamw":
        optimizer = adamw.AdamW(model.parameters(), lr=config.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    # weight = torch.tensor([1, 1.5, 1, 2, 1.5, 2, 2, 1.2]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=weight)

    criterion = nn.CrossEntropyLoss()

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 30, 35, 40], gamma=0.5)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=5, verbose=True)
    scheduler1 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer1, T_0=15, eta_min=1e-4)
    scheduler2 = lr_scheduler.CosineAnnealingWarmRestarts(optimizer2, T_0=15, eta_min=1e-4)

    global_step = 0
    max_miou = 0
    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        cm = np.zeros([3, 3])
        print(optimizer2.param_groups[0]['lr'])
        with tqdm(total=config.num_train, desc="Epoch %d / %d" % (epoch + 1, config.num_epochs),
                  unit='img', ncols=100) as train_pbar:
            model.train()

            for image, mask in train_loader:
                image = image.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.float32)
                aux_out, out, items = model(image)

                if not config.multi_stage:
                    gt_points = sampling_features(mask, items['points'], mode='nearest', align_corners=ALIGN_CORNERS).argmax(dim=1)
                    rend = items['rend']
                    # print("\n")
                    # print((gt_points==0).sum(), (gt_points==1).sum(), (gt_points==2).sum())
                    # print((rend == 0).sum(), (rend == 1).sum(), (rend == 2).sum())
                    point_loss = F.cross_entropy(rend, gt_points)
                else:
                    stage3, stage4, stage5 = items.values()
                    rend3 = stage3[1]
                    gt_points3 = sampling_features(mask, stage3[0], mode='nearest', align_corners=ALIGN_CORNERS).argmax(dim=1)
                    point_loss3 = F.cross_entropy(rend3, gt_points3)
                    rend4 = stage4[1]
                    gt_points4 = sampling_features(mask, stage4[0], mode='nearest', align_corners=ALIGN_CORNERS).argmax(dim=1)
                    point_loss4 = F.cross_entropy(rend4, gt_points4)
                    rend5 = stage5[1]
                    gt_points5 = sampling_features(mask, stage5[0], mode='nearest', align_corners=ALIGN_CORNERS).argmax(dim=1)
                    point_loss5 = F.cross_entropy(rend5, gt_points5)
                    point_loss = point_loss3 + point_loss4 + point_loss5

                mask = mask.long().argmax(dim=1)
                aux_loss = criterion(aux_out, mask)
                seg_loss = criterion(out, mask)

                loss = aux_loss + seg_loss + point_loss
                # loss = point_loss

                epoch_loss += loss.item()

                writer.add_scalar('Loss/aux_out', aux_loss.item(), global_step)
                writer.add_scalar('Loss/out', seg_loss.item(), global_step)
                writer.add_scalar('Loss/point', point_loss.item(), global_step)
                writer.add_scalar('Loss/train', loss.item(), global_step)
                train_pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                train_pbar.update(image.shape[0])
                global_step += 1
                # if global_step > 10:
                #     break

            # scheduler.step()
            print("\ntraining epoch loss: " + str(epoch_loss / (float(config.num_train) / (float(config.batch_size)))))
            torch.cuda.empty_cache()
        val_loss = 0
        with torch.no_grad():
            with tqdm(total=config.num_val, desc="Epoch %d / %d validation round" % (epoch + 1, config.num_epochs),
                      unit='img', ncols=100) as val_pbar:
                model.eval()
                locker = 0
                for image, mask in val_loader:
                    image = image.to(device, dtype=torch.float32)
                    target = mask.to(device, dtype=torch.long).argmax(dim=1)
                    mask = mask.cpu().numpy()

                    aux_pred, pred, final = model(image)
                    final = final['fine']
                    val_loss += F.cross_entropy(final, target).item()
                    final = final.cpu().detach().numpy()
                    mask = semantic_to_mask(mask, labels)
                    final = semantic_to_mask(final, labels)
                    cm += get_confusion_matrix(mask, final, labels)

                    val_pbar.update(image.shape[0])
                    if locker == 25:
                        writer.add_images('mask_a/true', mask[2, :, :], epoch + 1, dataformats='HW')
                        writer.add_images('mask_a/pred', final[2, :, :], epoch + 1, dataformats='HW')
                        writer.add_images('mask_b/true', mask[3, :, :], epoch + 1, dataformats='HW')
                        writer.add_images('mask_b/pred', final[3, :, :], epoch + 1, dataformats='HW')
                    locker += 1

                miou = get_miou(cm)
                scheduler1.step()
                scheduler2.step()
                precision, recall = get_classification_report(cm)
                writer.add_scalar('precision_tumor/val', precision[1], epoch + 1)
                writer.add_scalar('precision_lympha/val', precision[2], epoch + 1)
                writer.add_scalar('recall_tumor/val', recall[1], epoch + 1)
                writer.add_scalar('recall_lympha/val', recall[2], epoch + 1)
                writer.add_scalar('f1_tumor/val', (2 * precision[1] * recall[1] / (precision[1] + recall[1])), epoch + 1)
                writer.add_scalar('f1_lympha/val', (2 * precision[2] * recall[2] / (precision[2] + recall[2])), epoch + 1)
                if (miou[1] + miou[2]) / 2 > max_miou:
                    if torch.__version__ == "1.6.0":
                        torch.save(model,
                                   config.result_path + "/%d_%s_%.4f.pth" % (epoch + 1, config.model_type, (miou[1] + miou[2]) / 2),
                                   _use_new_zipfile_serialization=False)
                    else:
                        torch.save(model,
                                   config.result_path + "/%d_%s_%.4f.pth" % (epoch + 1, config.model_type, (miou[1] + miou[2]) / 2))
                    max_miou = (miou[1] + miou[2]) / 2
                print("\n")
                print(miou)
                print("testing epoch loss: " + str(val_loss), "Foreground mIoU = %.4f" % ((miou[1] + miou[2]) / 2))
                writer.add_scalar('Foreground mIoU/val', (miou[1] + miou[2]) / 2, epoch + 1)
                writer.add_scalar('loss/val', val_loss, epoch + 1)
                for idx, name in enumerate(objects):
                    writer.add_scalar('iou/val' + name, miou[idx], epoch + 1)
                torch.cuda.empty_cache()
    writer.close()
    print("Training finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=384)

    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--model_type', type=str, default='NewModel', help='UNet/UNet++/RefineNet')
    parser.add_argument('--data_type', type=str, default='multi', help='single/multi')
    parser.add_argument('--loss', type=str, default='ce', help='ce/dice/mix')
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam/adamw')
    parser.add_argument('--iscontinue', type=str, default=True, help='true/false')
    parser.add_argument('--smooth', type=str, default=False, help='true/false')
    parser.add_argument('--multi_stage', type=str, default=True, help='true/false')
    parser.add_argument('--freeze', type=str, default=False, help='true/false')

    parser.add_argument('--train_img_dir', type=str, default="../data/NPC20_V1/train/image")
    parser.add_argument('--train_mask_dir', type=str, default="../data/NPC20_V1/train/mask")
    parser.add_argument('--val_img_dir', type=str, default="../data/NPC20_V1/val/image")
    parser.add_argument('--val_mask_dir', type=str, default="../data/NPC20_V1/val/mask")
    parser.add_argument('--num_train', type=int, default=7300, help="4800/1600")
    parser.add_argument('--num_val', type=int, default=1824, help="1200/400")
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--result_path', type=str, default='./exp')

    config = parser.parse_args()
    train_val(config)