import torch
from torch import nn
from torch.nn import functional as F

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, high_level_channels, low_level_channels, num_classes, upsample_scale):
        super(DeepLabV3PlusDecoder, self).__init__()

        self.high_level_channels = high_level_channels
        self.low_level_channels = low_level_channels
        self.num_classes = num_classes
        self.upsample_scale = upsample_scale

        self.classifier = nn.Sequential(
            nn.Conv2d(high_level_channels + low_level_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, low_level_features, high_level_features):
        seg = self.classifier( torch.cat( [ low_level_features, high_level_features ], dim=1 ) )
        return F.interpolate(seg, scale_factor=self.upsample_scale, mode='bilinear', align_corners=False)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)