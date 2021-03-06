import torch.nn as nn
import torch

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}

def add_vgg(cfg, in_channels=3, batch_norm=False):
    layers = []
    in_channels = in_channels
    relu = nn.ReLU(inplace=True)
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(v)
            if batch_norm:
                layers += [conv2d, bn, relu]
            else:
                layers += [conv2d, relu]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, in_channels, size=300):
    in_channels = in_channels
    layers = []
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=(1, 3)[flag], padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    if size == 512:
        conv1 = nn.Conv2d(in_channels, 128, kernel_size=1, stride=1)
        conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)
        layers += [conv1, conv2]
    return layers

vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}

class VGG(nn.Module):
    def __init__(self, cfg):
        super(VGG, self).__init__()
        size = cfg.INPUT.IMAGE_SIZE
        vgg_config = vgg_base[cfg[str(size)]]
        vgg_extra = extras_base[str(size)]

        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(vgg_extra, in_channels=1024, size=size))
        self.l2_norm = None
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def init_from_pretrain(self, state_dict):
        self.vgg.load_state_dict(state_dict)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        features.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        return tuple(features)
