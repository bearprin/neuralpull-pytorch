# NeuralPull_pytorch

This repository contains the pytorch implementation of paper. [Neural-Pull: Learning Signed Distance Functions from Point Clouds by Learning to Pull Space onto Surfaces](https://arxiv.org/abs/2011.13495).

<img src="https://cdn.jsdelivr.net/gh/wzxshhz123/img_md/20210609135321.png" alt="image-20210609135318167" style="zoom:30%;" />

<img src="https://cdn.jsdelivr.net/gh/wzxshhz123/img_md/20210609142906.png" alt="image-20210609142852575" style="zoom:33%;" />

## Requirements

- Pytorch
- Numpy
- scipy
- trimesh
- skimage
- tqdm
- knn_cuda https://github.com/unlimblue/KNN_CUDA

## Dataset

- Put your own pointcloud files in 'npy_data' folder, each pointcloud file in a separate npy file
  - Data will be processed on loading.
- Also put ground-truth mesh in 'mesh' folder for evaluation

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --name <experiment name>
```

Each experiment result will be saved in experiment/experiment name and log to tensorboard.