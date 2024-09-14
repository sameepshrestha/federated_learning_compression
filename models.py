import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderMiniBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0, max_pooling=True):
        super(EncoderMiniBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_prob) if dropout_prob > 0 else None
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) if max_pooling else None

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        if self.dropout:
            x = self.dropout(x)
        skip_connection = x
        if self.max_pool:
            x = self.max_pool(x)
        return x, skip_connection

class DecoderMiniBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderMiniBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)  # Concatenating the skip connection
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class UNetCompiled(nn.Module):
    def __init__(self, input_channels=3, n_filters=32, n_classes=3):
        super(UNetCompiled, self).__init__()

        self.encoder1 = EncoderMiniBlock(input_channels, n_filters)
        self.encoder2 = EncoderMiniBlock(n_filters, n_filters * 2)
        self.encoder3 = EncoderMiniBlock(n_filters * 2, n_filters * 4)
        self.encoder4 = EncoderMiniBlock(n_filters * 4, n_filters * 8, dropout_prob=0)
        self.encoder5 = EncoderMiniBlock(n_filters * 8, n_filters * 16, dropout_prob=0, max_pooling=False)

        self.decoder6 = DecoderMiniBlock(n_filters * 16, n_filters * 8)
        self.decoder7 = DecoderMiniBlock(n_filters * 8 * 2, n_filters * 4)
        self.decoder8 = DecoderMiniBlock(n_filters * 4 * 2, n_filters * 2)
        self.decoder9 = DecoderMiniBlock(n_filters * 2 * 2, n_filters)

        self.conv9 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        x5, skip5 = self.encoder5(x4)

        # Decoder
        x6 = self.decoder6(x5, skip4)
        x7 = self.decoder7(x6, skip3)
        x8 = self.decoder8(x7, skip2)
        x9 = self.decoder9(x8, skip1)

        x9 = F.relu(self.conv9(x9))
        output = self.conv10(x9)

        return output
