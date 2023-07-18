import torch.nn as nn
import torchvision

activation_map = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
}


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    return size_all_mb


# old ff dims: f_dims=[1000, 500, 250], dropout=0.5,
class CNNClassifier(nn.Module):
    name = "CNNClassifier"

    def __init__(self, in_channels, out_dim,
                 conv_ks=[(7, 3), (5, 3), (3, 3)], conv_cs=[8, 16, 128],
                 conv_ss=[1, 1, 1], conv_ps=[(2, 1), (1, 1), (1, 1)],
                 pool_ks=[(4, 2), (4, 2), (2, 2)], pool_ss=[(4, 2), (2, 2), (2, 2)],
                 ff_dims=[500, 250, 125], dropout=0.5,
                 pooling="max", activation="gelu"):
        super().__init__()
        # Store architecture sizes & strides
        self.conv_kernels = conv_ks
        self.conv_channels = conv_cs
        self.conv_strides = conv_ss
        self.conv_pads = conv_ps
        self.pool_ks = pool_ks
        self.pool_ss = pool_ss
        self.hidden_layers = ff_dims
        # Pooling type
        if pooling == "max":
            self.pooling = nn.MaxPool2d
        else:
            self.pooling = nn.AvgPool2d

        self.activation_fn = activation_map[activation]
        # Build conv layers
        conv_chan_list = [in_channels] + self.conv_channels

        self.convs = nn.ModuleList()
        for i in range(len(conv_ks)):
            self.convs.append(nn.Conv2d(conv_chan_list[i], conv_chan_list[i + 1],
                                        self.conv_kernels[i], stride=self.conv_strides[i],
                                        padding=self.conv_pads[i]))
            self.convs.append(self.activation_fn())
            self.convs.append(self.pooling(self.pool_ks[i], self.pool_ss[i]))
        # Build linear layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Dropout(dropout))
        self.fc.append(nn.LazyLinear(self.hidden_layers[0]))
        for i in range(len(ff_dims) - 1):
            self.fc.append(self.activation_fn())
            self.fc.append(nn.Dropout(dropout))
            self.fc.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
        self.fc.append(nn.Linear(self.hidden_layers[-1], out_dim))

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        if len(x.shape) > 3:
            x = x.view(x.shape[0], -1)
        else:
            x = x.view(-1)
        for layer in self.fc:
            x = layer(x)
        return x


class CNNClassifierLight(nn.Module):
    name = "CNNClassifierLight"

    def __init__(self, in_channels, out_dim,
                 conv_ks=[(5, 5), (3, 3)], conv_cs=[3, 8],
                 conv_ss=[1, 1], conv_ps=[(2, 1), (1, 1)],
                 pool_ks=[(4, 2), (2, 2)], pool_ss=[(4, 2), (2, 2)],
                 ff_dims=[250, 125], dropout=0.5,
                 pooling="max", activation="gelu"):
        super().__init__()
        # Store architecture sizes & strides
        self.conv_kernels = conv_ks
        self.conv_channels = conv_cs
        self.conv_strides = conv_ss
        self.conv_pads = conv_ps
        self.pool_ks = pool_ks
        self.pool_ss = pool_ss
        self.hidden_layers = ff_dims
        # Pooling type
        if pooling == "max":
            self.pooling = nn.MaxPool2d
        else:
            self.pooling = nn.AvgPool2d

        self.activation_fn = activation_map[activation]
        # Build conv layers
        conv_chan_list = [in_channels] + self.conv_channels

        self.convs = nn.ModuleList()
        for i in range(len(conv_ks)):
            self.convs.append(nn.Conv2d(conv_chan_list[i], conv_chan_list[i + 1],
                                        self.conv_kernels[i], stride=self.conv_strides[i],
                                        padding=self.conv_pads[i]))
            self.convs.append(self.activation_fn())
            self.convs.append(self.pooling(self.pool_ks[i], self.pool_ss[i]))
        # Build linear layers
        self.fc = nn.ModuleList()
        self.fc.append(nn.Dropout(dropout))
        self.fc.append(nn.LazyLinear(self.hidden_layers[0]))
        for i in range(len(ff_dims) - 1):
            self.fc.append(self.activation_fn())
            self.fc.append(nn.Dropout(dropout))
            self.fc.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i + 1]))
        self.fc.append(nn.Linear(self.hidden_layers[-1], out_dim))

    def forward(self, x):
        for layer in self.convs:
            x = layer(x)
        if len(x.shape) > 3:
            x = x.view(x.shape[0], -1)
        else:
            x = x.view(-1)
        for layer in self.fc:
            x = layer(x)
        return x


class EfficientNet(nn.Module):
    name = "EfficientNet"

    def __init__(self, out_dim, dropout=0.1):
        super(EfficientNet, self).__init__()
        self.model = torchvision.models.efficientnet_b0(
            weights=None, dropout=dropout, num_classes=out_dim
        )

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        return self.model(x)


class ShuffleNet(nn.Module):
    name = "ShuffleNet"

    def __init__(self, out_dim, dropout=0.1):
        super().__init__()
        self.model = torchvision.models.shufflenetv2._shufflenetv2(
            weights=None,
            num_classes=out_dim,
            # Repeats and out channels are defaults from pytorch implementation
            stages_repeats=[4, 8, 4],
            stages_out_channels=[24, 48, 96, 192, 1024],
            progress=False,
        )

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        return self.model(x)


class ResidualBlock(nn.Module):
    """From https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """From https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/"""
    name = "ResNet"

    def __init__(self, block=ResidualBlock, layers=[2, 2, 4, 2], out_dim=2):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, self.inplanes, layers[0], stride=1)
        self.layer1 = self._make_layer(block, self.inplanes * 2, layers[1], stride=2)
        self.layer2 = self._make_layer(block, self.inplanes * 4, layers[2], stride=2)
        self.layer3 = self._make_layer(block, self.inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.LazyLinear(512, out_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
