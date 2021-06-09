import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init


class Net(nn.Module):
    def __init__(self, shape_num=1):
        super(Net, self).__init__()
        self.feat = nn.Sequential(
            nn.Linear(shape_num, 128),
            nn.ReLU()
        )
        self.encoder = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU()
        )

        decoder = [nn.Linear(512, 512), nn.ReLU()] * 7
        self.decoder = nn.Sequential(
            nn.Linear(640, 512),
            nn.ReLU(),
            *decoder
        )
        self.sdf = nn.Sequential(
            nn.Linear(512, 1),
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(512))
                init.constant_(m.bias, 0.0)
        for m in self.sdf.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=2 * np.sqrt(np.pi) / np.sqrt(512), std=0.000001)
                init.constant_(m.bias, -0.5)

    def forward(self, feature, input_query_points):
        '''

        :param feature: (B, 5000, 1)
        :param input_query_points: (B, 5000, 3)
        :return:
        '''
        feature_f = self.feat(feature)  # (B, 5000, 128)
        code = self.encoder(input_query_points)  # (B, 5000, 512)
        # print(feature_f.shape)
        # print(code.shape)
        code = torch.relu(torch.cat((code, feature_f), dim=2))  # (B, 5000, 640)
        code = self.decoder(code)  # (B, 5000, 512)
        sdf = self.sdf(code)  # (B, 5000, 1)
        return sdf
