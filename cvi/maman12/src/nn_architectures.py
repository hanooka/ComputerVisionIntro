import torch.nn as nn

import torch.nn.functional as F


class my_vgg(nn.Module):
    def __init__(self, num_classes=8, in_dim=3):
        super(my_vgg, self).__init__()
        self.in_dim = in_dim

        # configuration to the repeated conv layers
        cfg = [
            (64, 2),  # conv1
            (128, 2),  # conv2
            (256, 3),  # conv3
            (512, 3),  # conv4
            (512, 3),  # conv5
        ]

        self.features = self._make_layers(cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 8 * 8, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.35),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.35),
            # nn.Linear(4096, num_classes),
            nn.Flatten(),  # Flatten after GAP
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),  # Final output layer
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_dim

        for out_channels, num_repeats in cfg:
            for _ in range(num_repeats):
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Pool after each block

        return nn.Sequential(*layers)

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.classifier(x)
        return x


class FastCNN(nn.Module):
    def __init__(self, input_h, input_w, num_classes):
        super(FastCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BasicCNN(nn.Module):
    def __init__(self, input_h, input_w, num_classes=8):
        super(BasicCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # Output: 32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x32x32
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        flattened_size = (input_h // 4) * (input_w // 4) * 64


        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten for FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class OverFitFastCNN(nn.Module):
    def __init__(self, input_h, input_w, num_classes):
        super(OverFitFastCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
