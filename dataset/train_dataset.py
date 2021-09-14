import os

import numpy as np
import torch
import torch.utils.data as data
import tqdm

import trimesh

from knn_cuda import KNN


class SequentialPointCloudRandomPatchSampler(data.sampler.Sampler):
    def __init__(self, data_source, shape_num=1):
        super().__init__(data_source)
        self.data_source = data_source
        self.shape_num = shape_num

    def __iter__(self):
        rt = torch.randint(0, self.data_source.near_pts.shape[1], (self.data_source.near_pts.shape[1],))
        iter_order = [(i, rt[j]) for i in range(self.shape_num) for j in range(self.data_source.near_pts.shape[1] - 1)]
        return iter(iter_order)

    def __len__(self):
        return self.data_source.near_pts.shape[0] * self.data_source.near_pts.shape[1]
        # return self.shape_num


class PointGenerateDataset(data.Dataset):
    def __init__(self, pts_npy_path, device='cuda', query_num=25, gt_pts_num=20000, batch_size=5000, debug=False):
        super(PointGenerateDataset, self).__init__()
        if not os.path.exists(pts_npy_path):
            print("Error")

        self.gt_pts_num = gt_pts_num

        file_path = sorted(os.listdir(pts_npy_path))
        near_pts = list()
        querys = list()
        shape_num = 0
        for file in file_path:
            file.strip()
            if os.path.splitext(file)[1] != '.npy':
                continue
            shape_num += 1
            # load pt
            pnts = np.load(os.path.join(pts_npy_path, file))
            # sample ground truth to 20000
            pnts = pnts[self._patch_sampling(pnts)]
            pnts_pt = torch.from_numpy(pnts).float().to(device)
            knn = KNN(k=50, transpose_mode=True)
            # query each point for sigma^2
            dist, _ = knn(pnts_pt.reshape(1, pnts_pt.shape[0], pnts_pt.shape[1]),
                          pnts_pt.reshape(1, pnts_pt.shape[0], pnts_pt.shape[1]))
            dist = dist.squeeze()
            sigmas = dist[:, -1].unsqueeze(1)
            # sample query point
            query_point = pnts_pt + sigmas * torch.normal(mean=0.0, std=1.0,
                                                          size=(query_num, pnts_pt.shape[0], pnts_pt.shape[1]),
                                                          device=device)
            query_point = query_point.reshape(-1, batch_size, 3)
            knn.k = 1
            for i in tqdm.trange(query_point.shape[0]):
                # get the most nearest point from gt_pts for query point
                _, indx = knn(pnts_pt.reshape(1, pnts_pt.shape[0], pnts_pt.shape[1]),
                              query_point[i].reshape(1, query_point[i].shape[0], query_point[i].shape[1]))
                indx = indx.squeeze()
                # add for each batch
                near_pts.append(pnts_pt[indx].cpu().numpy())
                querys.append(query_point[i].cpu().numpy())
        if debug:
            near = trimesh.PointCloud(vertices=near_pts[2], colors=np.ones((batch_size, 4)))
            near.export(os.path.join('experiment', 'near.ply'))
            q = trimesh.PointCloud(vertices=querys[2], colors=np.full((batch_size, 4), fill_value=0.4))
            q.export(os.path.join('experiment', 'query.ply'))

        near_pts = np.asarray(near_pts).reshape(shape_num, -1, batch_size, 3)  # SHAPE_NUM, 100, 5000, 3
        querys = np.asarray(querys).reshape(shape_num, -1, batch_size, 3)

        self.shape_num = shape_num
        self.near_pts = near_pts
        self.query = querys

    def _patch_sampling(self, patch_pts):
        if patch_pts.shape[0] > self.gt_pts_num:
            sample_index = np.random.choice(range(patch_pts.shape[0]), self.gt_pts_num, replace=False)
        else:
            sample_index = np.random.choice(range(patch_pts.shape[0]), self.gt_pts_num)
        return sample_index

    def __len__(self):
        return self.near_pts.shape[0] * self.near_pts[1]

    def __getitem__(self, index):
        # print(index)
        return torch.from_numpy(self.near_pts[index]).float(), torch.from_numpy(self.query[index]).float(), index


class PointDataset(data.Dataset):
    def __init__(self, pts_npz_path, batch_size=5000):
        """

        :param pts_npz_path: abs path of mesh
        """
        super(PointDataset, self).__init__()
        if not os.path.exists(pts_npz_path):
            print("Error")

        file_path = sorted(os.listdir(pts_npz_path))
        near_pts = list()
        query = list()
        shape_num = 0
        for file in file_path:
            file.strip()
            if os.path.splitext(file)[1] != '.npz':
                continue
            shape_num += 1
            load_data = np.load(os.path.join(pts_npz_path, file))
            point = np.asarray(load_data['sample_near']).reshape(-1, batch_size, 3)  # 160, 5000, 3
            sample = np.asarray(load_data['sample']).reshape(-1, batch_size, 3)  # 160, 5000, 3
            near_pts.append(point)
            query.append(sample)

        near_pts = np.asarray(near_pts).reshape(-1, 5000, 3)
        query = np.asarray(query).reshape(-1, 5000, 3)

        self.shape_num = shape_num
        self.near_pts = near_pts
        self.query = query

    def __len__(self):
        return self.near_pts.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.near_pts[index]).float(), torch.from_numpy(self.query[index]).float()
