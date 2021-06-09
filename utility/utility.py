import random
import json

import numpy as np
from scipy.spatial import cKDTree

import trimesh
import torch
import torch.backends.cudnn as cudnn


class RunningAverage:
    """
    Example:
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def same_seed(seed):
    """

    :param seed:
    :return:
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def save_dict_to_json(d, json_path):
    """
    :param d:
    :param json_path:
    :return:
    """
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def normalize_mesh_export(mesh: trimesh.Trimesh, file_out=None):
    bounds = mesh.extents
    if bounds.min() == 0.0:
        return

    # translate to origin
    translation = (mesh.bounds[0] + mesh.bounds[1]) * 0.5
    translation = trimesh.transformations.translation_matrix(direction=-translation)
    mesh.apply_transform(translation)

    # scale to unit cube
    scale = 1.0 / bounds.max()
    scale_trafo = trimesh.transformations.scale_matrix(factor=scale)
    mesh.apply_transform(scale_trafo)
    if file_out is not None:
        mesh.export(file_out)
    return mesh


def eval_reconstruct_gt_mesh(rec_mesh: trimesh.Trimesh, gt_mesh: trimesh.Trimesh, sample_num=10000):
    def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
        ''' Computes minimal distances of each point in points_src to points_tgt.

        Args:
            points_src (numpy array): source points
            normals_src (numpy array): source normals
            points_tgt (numpy array): target points
            normals_tgt (numpy array): target normals
        '''
        kdtree = cKDTree(points_tgt)
        dist, idx = kdtree.query(points_src)

        if normals_src is not None and normals_tgt is not None:
            normals_src = \
                normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
            normals_tgt = \
                normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

            #        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
            #        # Handle normals that point into wrong direction gracefully
            #        # (mostly due to mehtod not caring about this in generation)
            #        normals_dot_product = np.abs(normals_dot_product)

            normals_dot_product = np.abs(normals_tgt[idx] * normals_src)
            normals_dot_product = normals_dot_product.sum(axis=-1)
        else:
            normals_dot_product = np.array(
                [np.nan] * points_src.shape[0], dtype=np.float32)
        return dist, normals_dot_product

    def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness ** 2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        accuracy2 = accuracy ** 2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()
        # print(completeness,accuracy,completeness2,accuracy2)
        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        return normals_correctness, chamferL1, chamferL2

    # sample point for rec
    pts_rec, idx = rec_mesh.sample(sample_num, return_index=True)
    pts_rec = pts_rec.astype(np.float32)
    normals_rec = rec_mesh.face_normals[idx].astype(np.float)
    # sample point for gt
    pts_gt, idx_gt = gt_mesh.sample(sample_num, return_index=True)
    pts_gt = pts_gt.astype(np.float32)
    normal_gt = gt_mesh.face_normals[idx_gt].astype(np.float)
    return eval_pointcloud(pts_rec, pts_gt, normals_rec, normal_gt)
