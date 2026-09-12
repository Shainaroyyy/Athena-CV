import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 32):
        super().__init__()
        hidden_channels = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(F.adaptive_avg_pool2d(x, 1))))
        max_out = self.fc2(self.relu(self.fc1(F.adaptive_max_pool2d(x, 1))))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv1(cat))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 32, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DecoderBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


class ResUNetCBAM(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        
        # Stem
        stem_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        stem_bn = nn.BatchNorm2d(64)
        stem_relu = nn.ReLU(inplace=True)
        stem_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-152 layers: [3, 8, 36, 3] Bottleneck blocks
        layer1 = self._make_layer(64, 64, blocks=3, stride=1)
        layer2 = self._make_layer(256, 128, blocks=8, stride=2)
        layer3 = self._make_layer(512, 256, blocks=36, stride=2)
        layer4 = self._make_layer(1024, 512, blocks=3, stride=2)

        # Encoder sequential matching keys encoder.0 .. encoder.7
        self.encoder = nn.Sequential(
            stem_conv,     # encoder.0
            stem_bn,       # encoder.1
            stem_relu,     # encoder.2
            stem_maxpool,  # encoder.3
            layer1,        # encoder.4
            layer2,        # encoder.5
            layer3,        # encoder.6
            layer4         # encoder.7
        )

        # CBAM attention blocks
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)
        self.cbam4 = CBAM(1024)
        self.cbam_center = CBAM(2048)

        # Decoder blocks
        self.decoder4 = DecoderBlock(3072, 512)
        self.decoder3 = DecoderBlock(1024, 256)
        self.decoder2 = DecoderBlock(512, 128)
        self.decoder1 = DecoderBlock(192, 64)

        # Final output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def _make_layer(self, inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion)
            )

        layers = [Bottleneck(inplanes, planes, stride=stride, downsample=downsample)]
        inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass
        e0 = self.encoder[0](x)
        e1 = self.encoder[1](e0)
        stem_feat = self.encoder[2](e1) # 64 channels, H/2, W/2
        e3 = self.encoder[3](stem_feat) # MaxPool: H/4, W/4
        e4 = self.encoder[4](e3)        # layer1: 256 channels, H/4, W/4
        e5 = self.encoder[5](e4)        # layer2: 512 channels, H/8, W/8
        e6 = self.encoder[6](e5)        # layer3: 1024 channels, H/16, W/16
        e7 = self.encoder[7](e6)        # layer4 / bottleneck: 2048 channels, H/32, W/32

        # CBAM attention forward pass
        c_stem = self.cbam1(stem_feat)
        c_e4 = self.cbam2(e4)
        c_e5 = self.cbam3(e5)
        c_e6 = self.cbam4(e6)
        c_center = self.cbam_center(e7)

        # Decoder forward pass with upsampling and skip connection concatenation
        d4_up = F.interpolate(c_center, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.decoder4(torch.cat([d4_up, c_e6], dim=1))

        d3_up = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d3_up, c_e5], dim=1))

        d2_up = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.decoder2(torch.cat([d2_up, c_e4], dim=1))

        d1_up = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.decoder1(torch.cat([d1_up, c_stem], dim=1))

        out = F.interpolate(self.output(d1), scale_factor=2, mode='bilinear', align_corners=True)
        return out
