import torch
import torch.utils.data as data


class ValDataset(data.Dataset):
    def __init__(self, bd=0.55, resolution=128):
        super(ValDataset, self).__init__()
        shape = (resolution, resolution, resolution)
        vxs = torch.arange(-bd, bd, bd * 2 / resolution)
        vys = torch.arange(-bd, bd, bd * 2 / resolution)
        vzs = torch.arange(-bd, bd, bd * 2 / resolution)
        pxs = vxs.view(-1, 1, 1).expand(*shape).contiguous().view(resolution ** 3)
        pys = vys.view(1, -1, 1).expand(*shape).contiguous().view(resolution ** 3)
        pzs = vzs.view(1, 1, -1).expand(*shape).contiguous().view(resolution ** 3)
        self.p = torch.stack([pxs, pys, pzs], dim=1).reshape(resolution, resolution ** 2, 3)

    def __len__(self):
        return self.p.shape[0]

    def __getitem__(self, index):
        return self.p[index]
