import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os

import torchvision

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# =======================================================

#=================================
# Scene Fine-Grained Sensing Module
#=================================
class LocalFeature(nn.Module):
    def __init__(self, embed_dim=512, in_channels=64, dropout_r=0.1):
        super(LocalFeature, self).__init__()
        self.conv_filter = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.sfgs = SFGS(embed_dim, dropout_r=dropout_r)
    
    def forward(self, x):
        x = self.conv_filter(x)
        x = self.sfgs(x)
        return x

class SFGS(nn.Module):
    def __init__(self , embed_dim=512, dim = 32, dropout_r=0.1):
        super(SFGS,self).__init__()
        self.embed_dim = embed_dim
        self.dim = dim
        self.dropout_r = dropout_r
        self.use_relu = True

        self.conv2d_block_11 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16,16))
        )
        self.conv2d_block_33 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=3, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16))
        )
        self.conv2d_block_55 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=5, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 16))
        )

        self.fc = FC(self.embed_dim // 2, self.embed_dim , self.dropout_r, self.use_relu)

        self.wsa = WSA(embed_dim, num_dim=128, is_weighted=True, dropout_r=self.dropout_r)

    def forward(self, vl_fea):
        bs, dim, _, _ = vl_fea.size()

        vl_1 = self.conv2d_block_11(vl_fea).view(bs, dim, -1)
        vl_2 = self.conv2d_block_33(vl_fea).view(bs, dim, -1)
        vl_3 = self.conv2d_block_55(vl_fea).view(bs, dim * 2, -1)

        vl_depth = torch.cat([vl_1,vl_2,vl_3], dim=1)

        return self.wsa(self.fc(vl_depth)).mean(1)
        # return self.fc(vl_depth)
    
#============================
# Weighted Self Attention
#============================
class WSA(nn.Module):
    def __init__(self, embed_dim=512, num_dim=128, is_weighted=False, dropout_r=0.1):
        super(WSA, self).__init__()
        self.num_dim = num_dim
        self.embed_dim = embed_dim
        self.is_weighted = is_weighted
        self.dropout_r = dropout_r

        self.mhatt = MHAtt(embed_dim, self.dropout_r)
        self.ffn = FeedForward(self.embed_dim, self.embed_dim * 2)

        self.dropout1 = nn.Dropout(self.dropout_r)
        self.norm1 = nn.LayerNorm(self.embed_dim)

        self.dropout2 = nn.Dropout(self.dropout_r)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # Learnable weights
        if is_weighted:
            self.fmp_weight = nn.Parameter(torch.randn(1, self.num_dim, self.embed_dim))
    def forward(self, x, x_mask=None):
        bs = x.shape[0]

        x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))

        if self.is_weighted:
            # feature map fusion
            x = self.fmp_weight.expand(bs, x.shape[1], x.shape[2]).transpose(1, 2).bmm(x)

        return x

# =======================================================

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=51, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dropout_r=0.1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.loc_fea1 = LocalFeature(dropout_r=dropout_r)
        # self.fc = nn.Linear(512 * block.expansion, num_classes) # baseline

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x0 = self.maxpool(x)
        local_fea1 = self.loc_fea1(x0)
        x1 = self.layer1(x0) # [batch, 256, 56, 56]
        x2 = self.layer2(x1) # [batch, 512, 28, 28]
        x3 = self.layer3(x2) # [batch, 1024, 14, 14]
        x4 = self.layer4(x3) # [batch, 2048, 7, 7]
        x5 = self.avgpool(x4) # [100, 2048, 1, 1] 传这个出去
        return local_fea1, x5
    
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)

#         return x


def _resnet(arch, block, layers, pretrained, progress, dropout_r=0.1, **kwargs):
    model = ResNet(block, layers, dropout_r=dropout_r, **kwargs)
    return model



def resnet50(pretrained=False, progress=True, dropout_r=0.1, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, dropout_r,
                   **kwargs)

def resnet101(pretrained=False, progress=True, dropout_r=0.1, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, dropout_r,
                   **kwargs)


#==================
# Some Reuse Module
#==================
# full connection layer
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout
        self.use_relu = use_relu
        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x
    
# Feed Forward Nets
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# class SA(nn.Module):
#     def __init__(self, num_dim = 512, is_weighted = False, dropout_r=0.1):
#         super(SA, self).__init__()
#         self.is_weighted = is_weighted
#         self.dropout_r = dropout_r
#         self.embed_dim = num_dim

#         self.mhatt = MHAtt(self.embed_dim, self.dropout_r)
#         self.ffn = FeedForward(self.embed_dim, self.embed_dim * 2)

#         self.dropout1 = nn.Dropout(self.dropout_r)
#         self.norm1 = nn.LayerNorm(self.embed_dim)

#         self.dropout2 = nn.Dropout(self.dropout_r)
#         self.norm2 = nn.LayerNorm(self.embed_dim)

#     def forward(self, x, x_mask=None):
#         bs = x.shape[0]

#         x = self.norm1(x + self.dropout1(self.mhatt(x, x, x, x_mask)))
#         x = self.norm2(x + self.dropout2(self.ffn(x)))

#         return x
        
#======================
# Multi-Head Attention
#======================
class MHAtt(nn.Module):
    def __init__(self, embed_dim=512, dropout_r=0.1):
        super(MHAtt, self).__init__()
        self.embed_dim = embed_dim
        self.dropout_r = dropout_r
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_merge = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout = nn.Dropout(self.dropout_r)

    def forward(self, v, k, q, mask=None):
        bs = q.size(0)

        v = self.linear_v(v).view(bs, -1, 8, 64).transpose(1, 2)
        k = self.linear_k(k).view(bs, -1, 8, 64).transpose(1, 2)
        q = self.linear_q(q).view(bs, -1, 8, 64).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(bs, -1, self.embed_dim)

        atted = self.linear_merge(atted)

        return atted

    def att(self, k, q, v, mask=None):
        d_k = q.shape[-1]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = torch.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, v)