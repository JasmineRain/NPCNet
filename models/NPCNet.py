import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Parameter, Softmax
from models.RendPoint import sampling_points_v2, sampling_features
from collections import OrderedDict
from torchvision import models

__all__ = ["NPCNet"]

ALIGN_CORNERS = False


class ASPPBlock(nn.Module):
    def __init__(self, in_channel=2048, out_channel=256, os=8, norm_layer=nn.BatchNorm2d):
        '''
        :param in_channel: default 2048 for resnet101
        :param out_channel: default 256 for resnet101
        :param os: 16 or 8
        '''
        super(ASPPBlock, self).__init__()
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.gave_pool = nn.Sequential(
            OrderedDict([('gavg', nn.AdaptiveAvgPool2d(rates[0])),
                         ('conv0_1', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn0_1', norm_layer(out_channel)),
                         ('relu0_1', nn.ReLU(inplace=True))])
        )
        self.conv1_1 = nn.Sequential(
            OrderedDict([('conv0_2', nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn0_2', norm_layer(out_channel)),
                         ('relu0_2', nn.ReLU(inplace=True))])
        )
        self.aspp_bra1 = nn.Sequential(
            OrderedDict([('conv1_1', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[1], dilation=rates[1], bias=False)),
                         ('bn1_1', norm_layer(out_channel)),
                         ('relu1_1', nn.ReLU(inplace=True))])
        )
        self.aspp_bra2 = nn.Sequential(
            OrderedDict([('conv1_2', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[2], dilation=rates[2], bias=False)),
                         ('bn1_2', norm_layer(out_channel)),
                         ('relu1_2', nn.ReLU(inplace=True))])
        )
        self.aspp_bra3 = nn.Sequential(
            OrderedDict([('conv1_3', nn.Conv2d(in_channel, out_channel, kernel_size=3,
                                               padding=rates[3], dilation=rates[3], bias=False)),
                         ('bn1_3', norm_layer(out_channel)),
                         ('relu1_3', nn.ReLU(inplace=True))])
        )
        self.aspp_catdown = nn.Sequential(
            OrderedDict([('conv_down', nn.Conv2d(5 * out_channel, out_channel, kernel_size=1, bias=False)),
                         ('bn_down', norm_layer(out_channel)),
                         ('relu_down', nn.ReLU(inplace=True)),
                         # ('drop_out', nn.Dropout(.1))
                         ])
        )

    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), size[2:], mode='bilinear', align_corners=ALIGN_CORNERS)
        x = torch.cat([x1,
                       self.conv1_1(x),
                       self.aspp_bra1(x),
                       self.aspp_bra2(x),
                       self.aspp_bra3(x)], dim=1)
        x = self.aspp_catdown(x)
        return x


class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=8):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):

    def __init__(self, num_channels):

        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):

        batch_size, channel, a, b = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, a, b))

        return output_tensor


class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=3, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1)  # batch x hw x c
        probs = F.softmax(self.scale * probs, dim=2)  # batch x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=nn.BatchNorm2d):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            bn_type(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            bn_type(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            bn_type(self.key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            bn_type(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            bn_type(self.key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0,
                      bias=False),
            bn_type(self.in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=nn.BatchNorm2d):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=nn.BatchNorm2d):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            bn_type(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output


class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, pretrained=False, norm_layer=None):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=False, replace_stride_with_dilation=[0, 1, 1],
                                            norm_layer=norm_layer)
            if pretrained:
                self.backbone.load_state_dict(torch.load("./resnet50-19c8e357.pth", map_location='cpu'))
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=False, replace_stride_with_dilation=[0, 1, 1],
                                             norm_layer=norm_layer)
            if pretrained:
                self.backbone.load_state_dict(torch.load("./resnet101-5d3b4d8f.pth", map_location='cpu'))
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=False, replace_stride_with_dilation=[0, 1, 1],
                                             norm_layer=norm_layer)
            if pretrained:
                self.backbone.load_state_dict(torch.load("./resnet152-b121ed2d.pth", map_location='cpu'))
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        # c0 = x
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)
        # print(c1.shape, c2.shape, c3.shape, c4.shape)

        return c1, c2, c3, c4


class DANetHead(nn.Module):

    def __init__(self, in_channels):
        super(DANetHead, self).__init__()
        self.att = ChannelSpatialSELayer(num_channels=in_channels, reduction_ratio=4)

    def forward(self, x):
        x = self.att(x)
        return x


class PointHead(nn.Module):
    def __init__(self, in_c=515, num_classes=1, k=800, beta=0.95):
        super(PointHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels=in_c, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=256, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.k = k
        self.beta = beta

    def forward(self, x, feature, mask):

        if not self.training:
            return self.inference(x, feature, mask)

        num_points = 100
        points = sampling_points_v2(torch.softmax(mask, dim=1), num_points, self.k, self.beta)
        coarse = sampling_features(mask, points, align_corners=ALIGN_CORNERS)
        fine = sampling_features(feature, points, align_corners=ALIGN_CORNERS)
        feature_representation = torch.cat([coarse, fine], dim=1)
        rend = self.mlp(feature_representation)

        return {"rend": rend, "points": points, "coarse": mask}

    @torch.no_grad()
    def inference(self, x, feature, mask):

        num_points = 1024
        while mask.shape[-1] != x.shape[-1]:
            mask = F.interpolate(mask, scale_factor=2, mode="bilinear", align_corners=ALIGN_CORNERS)

            points_idx, points = sampling_points_v2(torch.softmax(mask, dim=1), num_points, training=self.training)

            coarse = sampling_features(mask, points, align_corners=ALIGN_CORNERS)
            fine = sampling_features(feature, points, align_corners=ALIGN_CORNERS)

            feature_representation = torch.cat([coarse, fine], dim=1)

            rend = self.mlp(feature_representation)

            B, C, H, W = mask.shape

            points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
            mask = (mask.reshape(B, C, -1)
                    .scatter_(2, points_idx, rend)
                    .view(B, C, H, W))

        return {"fine": mask}


# base
class BaseModel(BaseNet):
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, pretrained=False):
        super(BaseModel, self).__init__(nclass, backbone, norm_layer=norm_layer, pretrained=pretrained)
        self.seg = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
                                 norm_layer(512),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
                                 )

    def forward(self, x):
        _, _, _, c4 = self.base_forward(x)
        mask = self.seg(c4)
        return F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=ALIGN_CORNERS)


# base + PEM
class BasePEM(BaseNet):
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, pretrained=False):
        super(BasePEM, self).__init__(nclass, backbone, norm_layer=norm_layer, pretrained=pretrained)
        self.da_head = DANetHead(2048)
        self.seg = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
                                 norm_layer(512),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
                                 )

    def forward(self, x):
        _, _, _, c4 = self.base_forward(x)
        mask = self.seg(self.da_head(c4))
        return F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=ALIGN_CORNERS)


# base + SEM
class BaseSEM(BaseNet):

    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, pretrained=False):
        super(BaseSEM, self).__init__(nclass, backbone, norm_layer=norm_layer, pretrained=pretrained)
        self.aspp = ASPPBlock(in_channel=2048, out_channel=512, norm_layer=norm_layer, os=8)
        self.conv3x3_ocr = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                         norm_layer(512),
                                         nn.ReLU(inplace=True)
                                         )
        self.ocr_gather_head = SpatialGather_Module(3)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.aux_head = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                                      norm_layer(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
                                      )
        self.cls_head = nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        _, _, _, c4 = self.base_forward(x)
        feats = self.aspp(c4)
        out_aux = self.aux_head(feats)
        feats = self.conv3x3_ocr(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(feats)

        return F.interpolate(out_aux, size=x.size()[-2:], mode='bilinear', align_corners=ALIGN_CORNERS), F.interpolate(
            out, size=x.size()[-2:], mode='bilinear', align_corners=ALIGN_CORNERS)


# base + BEM
class BaseBEM(BaseNet):
    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, pretrained=False):
        super(BaseBEM, self).__init__(nclass, backbone, norm_layer=norm_layer, pretrained=pretrained)
        self.seg = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
                                 norm_layer(512),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
                                 )
        self.rend_head = RendNet(n_class=3)

    def forward(self, x):
        c1, c2, c3, c4 = self.base_forward(x)
        mask = self.seg(self.da_head(c4))
        final = self.rend_head(c1, c2, c3, out)
        return F.interpolate(mask, size=x.size()[-2:], mode='bilinear', align_corners=ALIGN_CORNERS), final


class MLP(nn.Module):
    def __init__(self, fc_dim_in, fc_dims, num_classes):
        super(MLP, self).__init__()
        self.fc_dim_in = fc_dim_in
        self.fc_dims = fc_dims
        self.fc_nums = len(self.fc_dims)
        self.num_classes = num_classes
        self.mlp = nn.Sequential()
        for i in range(self.fc_nums):
            self.fc_dim = self.fc_dims[i]
            self.mlp.add_module("fc{}".format(i + 1),
                                nn.Conv1d(in_channels=self.fc_dim_in, out_channels=self.fc_dim, kernel_size=1, stride=1,
                                          padding=0, bias=True))
            self.mlp.add_module("relu{}".format(i + 1),
                                nn.ReLU(inplace=True))
            self.fc_dim_in = self.fc_dim
        self.mlp.add_module("final_fc",
                            nn.Conv1d(in_channels=self.fc_dim_in, out_channels=self.num_classes, kernel_size=1,
                                      stride=1, padding=0))

    def forward(self, x):
        out = self.mlp(x)
        return out


class RendNet(nn.Module):
    def __init__(self, n_class):
        super(RendNet, self).__init__()
        self.mlp3 = MLP(fc_dim_in=1027, fc_dims=[256, 256, 256], num_classes=n_class)
        self.mlp2 = MLP(fc_dim_in=515, fc_dims=[256, 256, 256], num_classes=n_class)
        self.mlp1 = MLP(fc_dim_in=259, fc_dims=[256, 256, 256], num_classes=n_class)

    def forward(self, x1, x2, x3, coarse):
        if not self.training:
            return self.inference(x1, x2, x3, coarse)

        # coarse size: 64x64
        # temp3 size : 128x128
        temp3 = F.interpolate(coarse, scale_factor=2, mode='bilinear', align_corners=ALIGN_CORNERS)
        points3 = sampling_points_v2(torch.softmax(temp3, dim=1), N=200, k=800, beta=0.95)
        coarse_feature = sampling_features(temp3, points3, align_corners=ALIGN_CORNERS)
        fine_feature = sampling_features(x3, points3, align_corners=ALIGN_CORNERS)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend3 = self.mlp3(feature_representation)

        # temp3 size : 256x256
        temp4 = F.interpolate(temp3, scale_factor=2, mode='bilinear', align_corners=ALIGN_CORNERS)
        points4 = sampling_points_v2(torch.softmax(temp4, dim=1), N=400, k=800, beta=0.95)
        coarse_feature = sampling_features(temp4, points4, align_corners=ALIGN_CORNERS)
        fine_feature = sampling_features(x2, points4, align_corners=ALIGN_CORNERS)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend4 = self.mlp2(feature_representation)

        # temp3 size : 512x512
        temp5 = F.interpolate(temp4, scale_factor=2, mode='bilinear', align_corners=ALIGN_CORNERS)
        points5 = sampling_points_v2(torch.softmax(temp5, dim=1), N=800, k=800, beta=0.95)
        coarse_feature = sampling_features(temp5, points5, align_corners=ALIGN_CORNERS)
        fine_feature = sampling_features(x1, points5, align_corners=ALIGN_CORNERS)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend5 = self.mlp1(feature_representation)

        return {
            "stage3": [points3, rend3],
            "stage4": [points4, rend4],
            "stage5": [points5, rend5],
        }

    @torch.no_grad()
    def inference(self, x1, x2, x3, coarse):
        # 128x128
        coarse3 = F.interpolate(coarse, scale_factor=2, mode='bilinear', align_corners=ALIGN_CORNERS)
        temp = coarse3
        points_idx, points = sampling_points_v2(torch.softmax(temp, dim=1), 1024, training=self.training)
        coarse_feature = sampling_features(temp, points, align_corners=ALIGN_CORNERS)
        fine_feature = sampling_features(x3, points, align_corners=ALIGN_CORNERS)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend = self.mlp3(feature_representation)
        B, C, H, W = coarse3.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        coarse3 = (coarse3.reshape(B, C, -1)
                   .scatter_(2, points_idx, rend)
                   .view(B, C, H, W))

        # 256x256
        coarse4 = F.interpolate(coarse3, scale_factor=2, mode='bilinear', align_corners=ALIGN_CORNERS)
        temp = coarse4
        points_idx, points = sampling_points_v2(torch.softmax(temp, dim=1), 1024, training=self.training)
        coarse_feature = sampling_features(temp, points, align_corners=ALIGN_CORNERS)
        fine_feature = sampling_features(x2, points, align_corners=ALIGN_CORNERS)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend = self.mlp2(feature_representation)
        B, C, H, W = coarse4.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        coarse4 = (coarse4.reshape(B, C, -1)
                   .scatter_(2, points_idx, rend)
                   .view(B, C, H, W))

        # 512x512
        coarse5 = F.interpolate(coarse4, scale_factor=2, mode='bilinear', align_corners=ALIGN_CORNERS)
        temp = coarse5
        points_idx, points = sampling_points_v2(torch.softmax(temp, dim=1), 1024, training=self.training)
        coarse_feature = sampling_features(temp, points, align_corners=ALIGN_CORNERS)
        fine_feature = sampling_features(x1, points, align_corners=ALIGN_CORNERS)
        feature_representation = torch.cat([coarse_feature, fine_feature], dim=1)
        rend = self.mlp1(feature_representation)
        B, C, H, W = coarse5.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        coarse5 = (coarse5.reshape(B, C, -1)
                   .scatter_(2, points_idx, rend)
                   .view(B, C, H, W))

        return {
            "fine": coarse5
        }


# # base + PEM + SEM + BEM
class NPCNet(BaseNet):

    def __init__(self, nclass, backbone, norm_layer=nn.BatchNorm2d, pretrained=False):
        super(NPCNet, self).__init__(nclass, backbone, norm_layer=norm_layer, pretrained=pretrained)
        self.da_head = DANetHead(2048)
        self.aspp = ASPPBlock(in_channel=2048, out_channel=512, norm_layer=norm_layer, os=8)
        self.conv3x3_ocr = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                                         norm_layer(512),
                                         nn.ReLU(inplace=True)
                                         )
        self.ocr_gather_head = SpatialGather_Module(3)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.aux_head = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                                      norm_layer(512),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
                                      )
        self.cls_head = nn.Conv2d(512, nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.rend_head = RendNet(n_class=3)

    def forward(self, x):
        c1, c2, c3, c4 = self.base_forward(x)
        feats = self.aspp(self.da_head(c4))
        out_aux = self.aux_head(feats)
        feats = self.conv3x3_ocr(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(feats)
        final = self.rend_head(c1, c2, c3, out)

        return F.interpolate(out_aux, size=x.size()[-2:], mode='bilinear', align_corners=ALIGN_CORNERS), \
               F.interpolate(out, size=x.size()[-2:], mode='bilinear', align_corners=ALIGN_CORNERS), \
               final


if __name__ == "__main__":
    net = NPCNet(backbone='resnet101', nclass=3, pretrained=False)
    img = torch.rand(2, 3, 512, 512)
    mask = torch.rand(2, 3, 512, 512)
    net.train()
    aux, out, final = net(img)
    print(aux.shape, out.shape)
    # for k, v in output.items():
    #     print(k, v.shape)
    # test = sampling_features(mask, output['points'], mode='nearest')
    # print(test.shape)
