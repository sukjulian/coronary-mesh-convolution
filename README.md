# Coronary mesh convolution
![architecture](img/pipeline.jpg)

This repository contains code accompanying our (MICCAI 2021) "Workshop on Statistical Atlases and Computational Modelling of the Heart" (STACOM) paper ["Mesh convolutional neural networks for wall shear stress estimation in 3D artery models"](https://link.springer.com/chapter/10.1007/978-3-030-93722-5_11). For questions, feel free to [contact me](mailto:j.m.suk@utwente.nl).

## Dependencies & packages
Dependencies:
* Python (tested on 3.9.13)
* PyTorch (tested on 1.12.1)
* PyTorch Geometric "PyG" (tested on 2.0.3) with
  * torch-cluster (tested on 1.6.0)
  * torch-scatter (tested on 2.0.9)
  * torch-sparse (tested on 0.6.15)

Packages:
```
pip install prettytable vtk trimesh potpourri3d tensorboard h5py robust_laplacian
```

You can install all of these with the provided conda environment file (CUDA 11.6):
```
conda env create -f environment.yml -n cmc
conda activate cmc
```

Additionally, we need gauge-equivariant mesh convolution:
```
git clone https://github.com/Qualcomm-AI-research/gauge-equivariant-mesh-cnn.git
cd gauge-equivariant-mesh-cnn
pip install .
```

## Data
You can download the dataset(s) from [here](https://drive.google.com/drive/folders/18lNjZPYKLmd7w-UX7GwepHAy2R-3YP3W?usp=sharing). The physical units for wall shear stress are [dyn/cm^2] = 0.1 [Pa].

We adapt the dataset-directory structure [used by PyTorch Geometric ("PyG")](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html). The directory with the dataset should contain a folder `raw` with the unprocessed data. Pre-processing creates a folder `processed` with the transformed data.
```
vessel-datasets
└── stead
    ├── bifurcating
    │   └── raw
    │       └── database.hdf5
    └── single
        └── raw
            └── database.hdf5
```

## Usage
Experiments are run by executing e.g. (options listed in `main.py`)
```
python main.py --model gem_gcn --artery_type single
```
and produce visualised output in the `vis` directory which can be viewed with e.g. [ParaView](https://www.paraview.org/).

Hyperparameters for neural network training are set in an experiment file, e.g. `exp/gem_gcn/stead.py`.

Training curves can be viewed with TensorBoard for PyTorch via
```
tensorboard --logdir=runs
```
We support parallelisation over multiple GPUs. Just use the command line option with a space-separated list.
```
python main.py --model stead --artery_type single --num_epochs 100 --gpu 0 1
```

## Network layout
![architecture](img/architecture.jpg)
This repository implements a three-scale mesh-based graph convolutional residual neural network with gauge-equivariant convolution. For details refer to our paper ["Mesh convolutional neural networks for wall shear stress estimation in 3D artery models"](https://arxiv.org/abs/2109.04797).

## DiffusionNet
We have included [DiffusionNet](https://arxiv.org/abs/2012.00888) as an additional baseline. The code is copy & pasted from [this](https://github.com/nmwsharp/diffusion-net) excellent repository.

## Publication
If you found this repository useful, please consider citing our paper:
```
@inproceedings{Suk/et/al/2021,
  author = {Julian Suk and Pim de Haan and Phillip Lippe and Christoph Brune and Jelmer M. Wolterink},
  title = {Mesh convolutional neural networks for wall shear stress estimation in 3D artery models},
  booktitle = {Statistical Atlases and Computational Models of the Heart},
  year = {2021}
}
```
