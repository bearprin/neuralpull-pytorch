import argparse
import os

import numpy as np

import skimage.measure as measure
import trimesh

import tqdm

# import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torch.optim import Adam

from utility import same_seed, normalize_mesh_export, eval_reconstruct_gt_mesh
from network import Net
from dataset import ValDataset, SequentialPointCloudRandomPatchSampler, PointGenerateDataset

parser = argparse.ArgumentParser()
parser.add_argument('--mesh_path', type=str, default='data/humanBodySeg/meshes')
parser.add_argument('--device', type=str, default='cuda')

parser.add_argument('--bd', type=float, default=0.55)
parser.add_argument('--resolution', type=int, default=128)
parser.add_argument('--thresholds', type=list, default=[0.0, 0.01, 0.05])

parser.add_argument('--name', type=str, default='base')
parser.add_argument('--epochs', type=int, default=40000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--num_workers', type=int, default=20)
parser.add_argument('--seed', type=int, default=40938661)

if __name__ == '__main__':
    # init
    args = parser.parse_args()
    same_seed(args.seed)
    # dataset
    print('Load dataset')
    train_ds = PointGenerateDataset('npy_data', device=args.device)
    train_sampler = SequentialPointCloudRandomPatchSampler(data_source=train_ds, shape_num=train_ds.shape_num)
    train_dl = data.DataLoader(train_ds, batch_size=1, num_workers=args.num_workers,
                               sampler=train_sampler, pin_memory=args.device == 'cuda',
                               )

    val_ds = ValDataset(bd=args.bd, resolution=args.resolution)
    val_dl = data.DataLoader(val_ds, batch_size=1, num_workers=args.num_workers, pin_memory=args.device == 'cuda')
    args.shape_num = train_ds.shape_num
    # network
    print('Network Configure')
    net = Net(train_ds.shape_num).to(args.device)
    criterion = nn.MSELoss().to(args.device)
    optimizer = Adam(net.parameters(), lr=args.lr)

    if not os.path.exists(os.path.join('experiment', args.name)):
        os.makedirs(os.path.join('experiment', args.name))

    # wandb.init(project='neural-pull', entity='wzxshhz123', name=args.name, config=args, sync_tensorboard=True)
    # wandb.watch(net, log='all')
    writer = SummaryWriter(os.path.join('experiment', args.name, 'logdir'))
    min_loss = np.inf
    min_cdl1 = np.inf
    for epoch in tqdm.trange(1, args.epochs):
        net.train()
        loss_sum = 0
        with tqdm.tqdm(total=len(train_dl), desc='train loop') as tq:
            for i, (t, q, ind) in enumerate(train_dl):
                t = t.to(args.device)
                q = q.to(args.device)

                q.requires_grad = True

                feat = torch.zeros((1, q.shape[1], args.shape_num)).to(args.device)
                feat[0, :, ind[0]] = 1
                sdf = net(feat, q)
                sdf.sum().backward(retain_graph=True)
                grad = q.grad.detach()
                grad = F.normalize(grad, dim=2)
                t_pre = q - grad * sdf

                optimizer.zero_grad()
                loss = criterion(t, t_pre)
                loss.backward()
                optimizer.step()

                # update loss
                loss_sum += loss.detach().cpu().item()

                tq.set_postfix(loss='{:06.5f}'.format(loss_sum))
                tq.update()
        with torch.no_grad():
            net.eval()
            gt_mesh = os.listdir('mesh')
            gt_mesh = sorted([x.strip() for x in gt_mesh if os.path.splitext(x)[1] in ['.obj', '.ply', '.off', '.stl']])
            nc = [0] * len(args.thresholds)
            cd_l1 = [0] * len(args.thresholds)
            cd_l2 = [0] * len(args.thresholds)
            mesh_dict = dict()
            for thresh_ind, thresh in enumerate(args.thresholds):
                mesh_dict[thresh_ind] = list()
            for shape_ind in range(args.shape_num):
                vox = list()
                for i, q in enumerate(val_dl):
                    q = q.to(args.device)
                    feat = torch.zeros((1, q.shape[1], args.shape_num)).to(args.device)
                    feat[0, :, shape_ind] = 1
                    sdf = net(feat, q)
                    vox.append(sdf.detach().cpu().numpy())
                # export each mesh
                vox = np.asarray(vox).reshape((args.resolution, args.resolution, args.resolution))
                vox_max = np.max(vox.reshape((-1)))
                vox_min = np.min(vox.reshape((-1)))
                # sdf in data range?
                if np.min(args.thresholds) < vox_min or np.min(args.thresholds) > vox_max:
                    continue
                for thresh_ind, thresh in enumerate(args.thresholds):
                    # if thresh < vox.min() or thresh > vox.max():
                    #     continue
                    # if np.sum(vox > 0.0) < np.sum(vox < 0.0):
                    #     thresh = -thresh
                    vertices, faces, _, _ = measure.marching_cubes(vox, thresh)

                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                    mesh = normalize_mesh_export(mesh)
                    nc_ind, cd_l1_ind, cd_l2_ind = eval_reconstruct_gt_mesh(rec_mesh=mesh,
                                                                            gt_mesh=trimesh.load_mesh(
                                                                                os.path.join('mesh', gt_mesh[
                                                                                    shape_ind]), process=False))
                    nc[thresh_ind] += nc_ind
                    cd_l1[thresh_ind] += cd_l1_ind
                    cd_l2[thresh_ind] += cd_l2_ind
                    mesh_dict[thresh_ind].append(mesh)

                    vertices_tensor = torch.from_numpy(mesh.vertices.copy()).float().reshape(1, vertices.shape[0],
                                                                                             vertices.shape[1])
                    faces_tensor = torch.from_numpy(mesh.faces.copy()).float().reshape(1, faces.shape[0],
                                                                                       faces.shape[1])
                    writer.add_mesh(str(shape_ind) + '_' + str(thresh), vertices=vertices_tensor, faces=faces_tensor,
                                    global_step=epoch)

            writer.add_scalar('loss', loss_sum / train_ds.shape_num, global_step=epoch)
            for thresh_ind, thresh in enumerate(args.thresholds):
                writer.add_scalar('metric/avg_nc/thresh_{}'.format(thresh), nc[thresh_ind] / train_ds.shape_num,
                                  global_step=epoch)
                writer.add_scalar('metric/avg_cdl1/thresh_{}'.format(thresh), cd_l1[thresh_ind] / train_ds.shape_num,
                                  global_step=epoch)
                writer.add_scalar('metric/avg_cdl2/thresh_{}'.format(thresh), cd_l2[thresh_ind] / train_ds.shape_num,
                                  global_step=epoch)
                if min_cdl1 >= cd_l1[thresh_ind] / train_ds.shape_num:
                    min_cdl1 = cd_l1[thresh_ind] / train_ds.shape_num
                    print('normals_correctness:', nc[thresh_ind] / train_ds.shape_num,
                          'chamferL1:', cd_l1[thresh_ind] / train_ds.shape_num,
                          'chamferL2:', cd_l2[thresh_ind] / train_ds.shape_num)
                    for shape_ind, m in enumerate(mesh_dict[thresh_ind]):
                        m.export(os.path.join('experiment', args.name,
                                              str(epoch) + '_cd_l1_' + str(
                                                  min_cdl1) + '_' + str(
                                                  shape_ind) + '_' + str(
                                                  thresh) + '.obj'))
            # save ?
            if min_loss >= loss_sum / train_ds.shape_num:
                min_loss = loss_sum / train_ds.shape_num
                torch.save(net.state_dict(), os.path.join('experiment', args.name,
                                                          str(epoch) + '_loss_' + str(min_loss) + '.pth'))
                print('log at epoch: {}, min_loss: {}'.format(epoch, min_loss))
                for thresh_ind, thresh in enumerate(args.thresholds):
                    for shape_ind, m in enumerate(mesh_dict[thresh_ind]):
                        normalize_mesh_export(mesh,
                                              os.path.join('experiment', args.name,
                                                           str(epoch) + '_loss_' + str(
                                                               min_loss) + '_' + str(
                                                               shape_ind) + '_' + str(
                                                               thresh) + '.obj'))
    writer.close()
