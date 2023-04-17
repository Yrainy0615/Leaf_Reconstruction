import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class LineBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(LineBlock, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.linear(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_ch, num_feat=100):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(in_ch, 32, kernel=3, stride=2, pad=2)
        self.conv2 = ConvBlock(32, 64, kernel=3, stride=2, pad=2)
        self.conv3 = ConvBlock(64, 128, kernel=3, stride=2, pad=2)
        self.conv4 = ConvBlock(128, 256, kernel=3, stride=2, pad=2)
        self.linear1 = LineBlock(256*10*10, num_feat*2)
        self.linear2 = LineBlock(num_feat*2, num_feat)

    def forward(self, img):
        out_conv1 = self.conv1(img)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        out5 = out_conv4.view(out_conv4.shape[0], -1)
        out_linear1 = self.linear1(out5)
        out_linear2 = self.linear2(out_linear1)

        return out_linear2


class ShapeTextPredictor(nn.Module):
    def __init__(self, num_feat, num_vert):
        super(ShapeTextPredictor, self).__init__()
        self.pred_layer = nn.Linear(num_feat, num_vert * 6)

    def forward(self, feat):
        delta_v = self.pred_layer.forward(feat)
        delta_v = delta_v.view(delta_v.shape[0], -1, 6)
        verts = delta_v[:, :, :3]
        texture = delta_v[:, :, 3:]
        texture = nn.Sigmoid()(texture)

        return verts, texture


class ScalePredictor(nn.Module):
    def __init__(self, num_feat):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(num_feat, 1)

    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1
        scale = nn.ReLU()(scale) + 1e-12

        return scale


class MeshNet(nn.Module):
    def __init__(self, in_ch, num_feat, verts_shape):
        super(MeshNet, self).__init__()
        num_vert = verts_shape[0]
        self.encoder = Encoder(in_ch, num_feat=num_feat)
        self.mesh_predictor = ShapeTextPredictor(num_feat=num_feat, num_vert=num_vert)
        self.scale_predictor = ScalePredictor(num_feat=num_feat)

        # network initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)

        # mean deformation
        self.mean_deform = nn.Parameter(torch.zeros(verts_shape))

    def forward(self, img):
        feat = self.encoder(img)
        delta_v, texture = self.mesh_predictor(feat)
        scale = self.scale_predictor(feat)

        return delta_v, texture, scale

    def get_mean_deform(self):
        return self.mean_deform
