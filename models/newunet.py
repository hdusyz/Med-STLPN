import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, training=True):
        super(SegNet, self).__init__()
        self.training = training
        
        # Encoder layers: 3D convolutions with 1 stride and padding to keep spatial dimensions (height, width)
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 32, 16, 512, 512
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)         # b, 64, 16, 512, 512
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)        # b, 128, 16, 512, 512
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)       # b, 256, 16, 512, 512
        
        # Decoder layers
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 128, 16, 512, 512
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)   # b, 64, 16, 512, 512
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)    # b, 32, 16, 512, 512
        self.decoder5 = nn.Conv3d(32, 1, 3, stride=1, padding=1)     # b, 2, 16, 512, 512
        
        # Mapping layers for multi-scale outputs
        self.map4 = nn.Sequential(
            nn.Conv3d(32, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(2, 4, 4), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        self.map1 = nn.Sequential(
            nn.Conv3d(1, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))  # b, 32, 8, 256, 256
        #print(out.shape)
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))  # b, 64, 4, 128, 128
        #print(out.shape)
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))  # b, 128, 2, 64, 64
        #print(out.shape)
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))  # b, 256, 1, 32, 32
        #print(out.shape)

        # Decoder
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))  # b, 128, 2, 64, 64
        #print(out.shape)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))  # b, 64, 4, 128, 128
        #print(out.shape)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))  # b, 32, 8, 256, 256
        #print(out.shape)
        output4 = self.map4(out)
        #print(out.shape)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))  # b, 2, 16, 512, 512
        #print(out.shape)
        # Final output
        #output1 = self.map1(out)
        
        return out