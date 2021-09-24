# neuralpull-pytorch

This repository is an unofficial Pytorch implementation of [Neural-Pull: Learning Signed Distance Functions from Point Clouds by Learning to Pull Space onto Surfaces](https://arxiv.org/abs/2011.13495).

<img src="./img/bunny.png" alt="bunny">

## Quick Start
```python
# clone this repo
git clone https://github.com/bearprin/NeuralPull_pytorch.git

# create a conda environment 
conda env create -f env.yaml

# activate the new conda environment
conda activate neural-pull

# train and evaluate the with default settings
python train.py
```

## Dataset

- Put your own pointcloud files in 'npy_data' folder, **each pointcloud file in a separate .npy file**
  - Data will be processed on loading.
- Also put ground-truth mesh in 'mesh' folder for evaluation

## Training

To train the model, run this command:

```train
python train.py --name <experiment name>
```

Each experiment result will be saved in experiment/experiment name and log evaluation to tensorboard.

