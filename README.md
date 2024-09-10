_As of 28 June 2023,_ [`gem_cnn`](https://github.com/Qualcomm-AI-research/gauge-equivariant-mesh-cnn) _is implemented more efficiently with ca. 60 % speed-up, so make sure to re-install!_

_This repository contains code for learning on surface meshes. If you are instead trying to learn on volume meshes, try_ [this one](https://github.com/sukjulian/segnn-hemodynamics)_._

# Coronary mesh convolution
![architecture](img/pipeline.jpg)

This repository contains the official implementation of ["Mesh convolutional neural networks for wall shear stress estimation in 3D artery models"](https://link.springer.com/chapter/10.1007/978-3-030-93722-5_11) (STACOM workshop @ MICCAI 2021) and ["Mesh neural networks for SE(3)-equivariant hemodynamics estimation on the artery wall"](https://www.sciencedirect.com/science/article/pii/S0010482524004128) (Computers in Biology and Medicine). For questions, feel free to [contact me](mailto:j.m.suk@utwente.nl).

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
If you get an error regarding OpenMesh, try
```
conda install -c conda-forge openmesh-python
```
and then try to install again.

## Data
You can download the dataset(s) from [here](https://drive.google.com/drive/folders/18lNjZPYKLmd7w-UX7GwepHAy2R-3YP3W?usp=sharing). The physical units for wall shear stress are [dyn/cm^2] = 0.1 [Pa]. We additionally provide [pre-trained model weights](https://drive.google.com/drive/folders/1o-vklPaGulkpLkM7TiwBmVAAN4vvpaJf?usp=sharing).

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
The pre-trained model weights should be placed in a folder `model-weights` and are loaded automatically if present.

## Usage
Experiments are run by executing e.g. (options listed in `main.py`)
```
python main.py --model gem_gcn --artery_type single
```
and produce visualised output in the `vis` directory which can be viewed with e.g. [ParaView](https://www.paraview.org/). If you get an error `Unable to open file` try downloading the HDF5 files directly instead of the whole directory and placing them in their respective folders manually. If everything works, first thing you will see is the pre-processing of the training data.

Hyperparameters for neural network training are set in an experiment file, e.g. `exp/gem_gcn/stead.py`. Training curves can be viewed with TensorBoard for PyTorch via
```
tensorboard --logdir=runs
```
This codebase supports parallelisation over multiple GPUs. Just use the command line option with a space-separated list.
```
python main.py --model gem_gcn --artery_type single --num_epochs 100 --gpu 0 1
```

## Network layout
![architecture](img/architecture.jpg)
This repository implements a three-scale mesh-based graph convolutional residual neural network with gauge-equivariant convolution. For details refer to our paper ["Mesh convolutional neural networks for wall shear stress estimation in 3D artery models"](https://arxiv.org/abs/2109.04797).

## DiffusionNet
We have included [DiffusionNet](https://arxiv.org/abs/2012.00888) as an additional baseline. The code is copy & pasted from [this](https://github.com/nmwsharp/diffusion-net) excellent repository.

## Publications
If you found this repository useful, please consider citing our paper(s):
```
@article{SUK2024108328,
title = {Mesh neural networks for SE(3)-equivariant hemodynamics estimation on the artery wall},
journal = {Computers in Biology and Medicine},
volume = {173},
pages = {108328},
year = {2024},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2024.108328},
url = {https://www.sciencedirect.com/science/article/pii/S0010482524004128},
author = {Julian Suk and Pim {de Haan} and Phillip Lippe and Christoph Brune and Jelmer M. Wolterink},
}

@InProceedings{10.1007/978-3-030-93722-5_11,
author="Suk, Julian and Haan, Pim de and Lippe, Phillip and Brune, Christoph and Wolterink, Jelmer M.",
editor="Puyol Ant{\'o}n, Esther and Pop, Mihaela and Mart{\'i}n-Isla, Carlos and Sermesant, Maxime and Suinesiaputra, Avan and Camara, Oscar and Lekadir, Karim and Young, Alistair",
title="Mesh Convolutional Neural Networks for Wall Shear Stress Estimation in 3D Artery Models",
booktitle="Statistical Atlases and Computational Models of the Heart. Multi-Disease, Multi-View, and Multi-Center Right Ventricular Segmentation in Cardiac MRI Challenge",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="93--102",
isbn="978-3-030-93722-5"
}
```
