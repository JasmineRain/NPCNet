# 相比v5增加了dense concat


import torch.nn as nn
import torch
import torch.nn.functional as F
import copy as cp
import gc
from torchvision import models
from utils_Deeplab import SyncBN2d

# from pretrainedmodels import resnet50
#
# from config import *


__all__ = ['RDSN']


norm_layer = SyncBN2d


class SEModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        # x = self.rReLU(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Conv_Unit(nn.Module):
    def __init__(self, in_channels, out_channels, se=False, max_pool=False):
        super(Conv_Unit, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), norm_layer(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, 3, padding=1), norm_layer(out_channels),
                                  nn.ReLU(inplace=True))
        self.max_pool = max_pool
        self.se = se
        if max_pool:
            self.pool = nn.MaxPool2d(2, 2)
        if se:
            self.SEBlock = SEModule(out_channels)
        # self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        out = self.conv(x)
        if self.max_pool:
            out = self.pool(out)
        if self.se:
            out = self.SEBlock(out)
        # out = self.dropout(out)
        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.ReLU = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.ReLU(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)


class ASPP(nn.Module):
    def __init__(self, inplanes, out_channels, output_stride=16):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, out_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, out_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, out_channels, 1, stride=1, bias=False),
                                             # norm_layer(out_channels),
                                             nn.RReLU())
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.ReLU = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class RRB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.reduct_layer = nn.Conv2d(in_channels, out_channels, 1)
        self.up = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), norm_layer(out_channels),
                                nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.last = nn.Sequential(nn.ReLU(inplace=True))
        # , norm_layer(out_channels))
        self.dropout = nn.Dropout2d(0.5)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.reduct_layer(x)
        up = self.up(out)
        out = out + up
        out = self.last(out)
        # return self.dropout(out)
        return out


class RSEBlock(nn.Module):
    def __init__(self, channels):
        super(RSEBlock, self).__init__()

        self.channels = channels

        self.downs = nn.ModuleList([])

        for i in range(len(channels)):
            self.downs.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels[i], 128, 1),
                                            nn.ReLU(inplace=True)))
        self.classifier = nn.GRU(128, 128, 2, batch_first=True)

        self.reductions = nn.ModuleList([])

        for i in range(len(channels)):
            self.reductions.append(nn.Conv2d(128, channels[i], 1))

    def forward(self, features):
        self.classifier.flatten_parameters()
        fs = []

        for i in range(len(self.channels)):
            fs.append(self.downs[i](features[i]).view(features[i].size(0), 1, -1))
        attention = torch.cat([f for f in fs], dim=1)
        attention, hc = self.classifier(attention)
        atts = attention.chunk(len(self.channels), dim=1)
        for i in range(len(self.channels)):
            att = self.reductions[i](atts[i].view(atts[i].size(0), -1, 1, 1))
            features[i] = features[i] * F.sigmoid(att)

        return features


class RSABlock(nn.Module):
    def __init__(self, channels, out_channels=64):
        super(RSABlock, self).__init__()
        self.channels = channels

        self.downs = nn.ModuleList([])
        for i in range(len(channels)):
            self.downs.append(nn.Sequential(
                nn.Conv2d(channels[i], 32, 1), nn.AdaptiveAvgPool2d((7, 7)))
            )
        self.spatial_attention = nn.GRU(32 * 7 * 7, 7 * 7, batch_first=True)

        self.reduction = nn.ModuleList([])
        for i in range(len(channels)):
            self.reduction.append(nn.Sequential(nn.Conv2d(1, channels[i], 3, padding=1), nn.Sigmoid()))
        self.out_blocks = nn.ModuleList([])
        for i in range(len(channels)):
            self.out_blocks.append(nn.Conv2d(channels[i], out_channels, 1))

    def forward(self, features):
        r_in = []
        self.spatial_attention.flatten_parameters()
        for i in range(len(self.channels)):
            r_in.append(self.downs[i](features[i]).view(features[i].size(0), 1, -1))
        r_in = torch.cat([f for f in r_in], dim=1)
        r_out, hc = self.spatial_attention(r_in)
        r_out = r_out.chunk(len(self.channels), dim=1)
        attentions = []
        for i in range(len(self.channels)):
            attentions.append(
                F.upsample(r_out[i].view(r_out[i].size(0), 1, 7, 7), size=features[i].size()[2:], mode='bilinear', align_corners=False))
            attentions[i] = self.reduction[i](attentions[i])
            attentions[i] = features[i] * attentions[i]
            attentions[i] = self.out_blocks[i](attentions[i])

        return attentions


class RDSN(nn.Module):
    def __init__(self, nclass=3):

        super(RDSN, self).__init__()

        pretrained_model = models.resnet101(pretrained=False, progress=True)
        pretrained_model.load_state_dict(torch.load("./resnet101-5d3b4d8f.pth", map_location='cpu'))
        self.layer0 = nn.Sequential(pretrained_model.conv1,
                                    pretrained_model.bn1,
                                    pretrained_model.relu
                                    )  # 64 channels
        self.layer1 = nn.Sequential(pretrained_model.maxpool,
                                    pretrained_model.layer1)  # 256 channels
        self.layer2 = pretrained_model.layer2  # 512 channels
        self.layer3 = pretrained_model.layer3  # 1024 channels
        # for p in self.parameters():
        #     p.requires_grad = False

        # del pretrained_model
        # gc.collect()
        # torch.cuda.empty_cache()

        self.refine = Conv_Unit(3, 64)

        self.RSE = RSEBlock([1024, 512, 256, 64, 64])

        self.reduct = nn.Sequential(nn.Conv2d(1024 + 512 + 256 + 64 + 64, 64, 1))

        # if SUPERVISION:
        #     self.supervision1 = nn.Conv2d(256, LAST_CHANNEL, 1)
        #     self.supervision2 = nn.Conv2d(512, LAST_CHANNEL, 1)
        #     self.supervision3 = nn.Conv2d(1024, LAST_CHANNEL, 1)

        self.refine3_1 = Conv_Unit(1024 * 2, 512)
        self.refine2_1 = Conv_Unit(512 * 2, 256)
        self.refine1_1 = Conv_Unit(256 * 2, 64)
        self.refine0_1 = Conv_Unit(64 * 2, 64)
        self.refine_1 = Conv_Unit(64 * 2, 64)

        self.spatial = RSABlock([512, 256, 64, 64], 64)

        self.GlobalAvgPooling = nn.AdaptiveAvgPool2d((1, 1))
        self.up2 = nn.Conv2d(512, 512, 3, padding=1)
        self.up1 = nn.Conv2d(256, 256, 3, padding=1)
        self.up0 = nn.Conv2d(64, 64, 3, padding=1)
        self.up = nn.Conv2d(64 * 4, 64, 3, padding=1)

        # self.final0 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, nclass, 1))
        # self.final1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, nclass, 1))
        # self.final2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, nclass, 1))
        # self.final3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, nclass, 1))
        self.final = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, nclass, 1))

        # if INIT_WEIGHT:
        #     self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        l = self.refine(x)

        [l3, l2, l1, l0, l] = self.RSE([l3, l2, l1, l0, l])

        G = self.GlobalAvgPooling(l3)

        bridge = F.upsample(G, size=l3.size()[2:], mode='bilinear', align_corners=False)
        l3_ = torch.cat((l3, bridge), dim=1)
        l3_ = self.refine3_1(l3_)

        bridge = self.up2(F.upsample(l3_, scale_factor=2, mode='bilinear', align_corners=False))
        l2_ = torch.cat((l2, bridge), dim=1)
        l2_ = self.refine2_1(l2_)

        bridge = self.up1(F.upsample(l2_, scale_factor=2, mode='bilinear', align_corners=False))
        l1_ = torch.cat((l1, bridge), dim=1)
        l1_ = self.refine1_1(l1_)

        bridge = self.up0(F.upsample(l1_, scale_factor=2, mode='bilinear', align_corners=False))
        l0_ = torch.cat((l0, bridge), dim=1)
        l0_ = self.refine0_1(l0_)

        features = self.spatial([l3_, l2_, l1_, l0_])
        for i in range(len(features)):
            features[i] = F.upsample(features[i], size=x.size()[-2:], mode='bilinear', align_corners=False)

        # l3_ = features[0]
        # l2_ = features[1]
        # l1_ = features[2]
        # l0_ = features[3]

        bridge = torch.cat([f for f in features], dim=1)
        bridge = self.up(bridge)
        l = torch.cat((l, F.upsample(l0, size=l.size()[2:], mode='bilinear', align_corners=False),
                       F.upsample(l1, size=l.size()[2:], mode='bilinear', align_corners=False),
                       F.upsample(l2, size=l.size()[2:], mode='bilinear', align_corners=False),
                       F.upsample(l3, size=l.size()[2:], mode='bilinear', align_corners=False)), dim=1)
        l = self.reduct(l)
        l = torch.cat((l, bridge), dim=1)
        l = self.refine_1(l)

        # predict0 = F.upsample(self.final0(l0_), CROP_SIZE, mode='bilinear')
        # predict1 = F.upsample(self.final1(l1_), CROP_SIZE, mode='bilinear')
        # predict2 = F.upsample(self.final2(l2_), CROP_SIZE, mode='bilinear')
        # predict3 = F.upsample(self.final3(l3_), CROP_SIZE, mode='bilinear')
        # predict0 = F.upsample(self.final(l0_), CROP_SIZE, mode='bilinear')
        predict = self.final(l)
        return predict

        # if self.training:
        #     # if LOSS != 'BCE':
        #     #     p1 = F.sigmoid(p1)
        #     #     p2 = F.sigmoid(p2)
        #     #     p3 = F.sigmoid(p3)
        #     #     p4 = F.sigmoid(p4)
        #     #     mask = F.sigmoid(mask)
        #
        #     # return [predict]
        #     if SUPERVISION:
        #         return [predict1, predict2, predict3, predict]
        #     else:
        #         # return [predict1, predict2, predict3, predict0, predict]
        #         return [predict1, predict3, predict]
        # else:
        #     # if LOSS != 'BCE':
        #     #     mask = F.sigmoid(mask)
        #     # weight = 4
        #     mask = predict
        #     return mask
        #     # return predict


if __name__ == '__main__':
    model = RDSN()
    # model = model.eval()
    img = torch.rand(1, 3, 256, 256)
    model.eval()
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    res = model(img)
    print(res.shape)
    exit(0)
