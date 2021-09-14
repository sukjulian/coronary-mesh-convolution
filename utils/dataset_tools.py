import os
import glob
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from datasets.multiscale import MultiscaleData as Data
import torch


def raw_file_names(dataset):
    root = os.path.join(dataset.root, "raw")
    absolute = glob.glob(os.path.join(root, dataset.pattern))
    absolute.sort()  # comparability
    absolute = absolute[dataset.split[0]:dataset.split[1]]

    return [os.path.relpath(a, root) for a in absolute]


def load_process(dataset, path):

    # Read the mesh data
    dataset.reader.SetFileName(path)
    dataset.reader.Update()
    mesh = dataset.reader.GetOutput()

    # Convert to NumPy
    vertices = vtk_to_numpy(mesh.GetPoints().GetData())  # float32
    labels = vtk_to_numpy(mesh.GetPointData().GetArray("vWSS")).astype(np.float32)  # float64
    '''
    "Array 2 name = vWSS ... Array 5 name = pressure ... Array 6 name = velocity"
    (if average values were computed)
    '''
    polygons = vtk_to_numpy(mesh.GetPolys().GetData())  # int64
    """
    The first elements in "polygons" specify how many nodes are in
    the current polygon, followed by the corresponding point indices,
    e.g. z = [3 0 1 2 3 3 4 5 3 6 ...]. Assume all are triangles:
    """
    polygons = polygons.reshape((int(polygons.shape[0] / 4), 4))
    polygons = polygons[:, 1:]
    polygons[:, [1, 0]] = polygons[:, [0, 1]]  # correct face orientation

    # Scalar labels, e.g. wall shear stress magnitude
    # labels = np.linalg.norm(labels, axis=1)

    # Load into PyTorch geometric
    data = Data(y=torch.from_numpy(labels).float(),
                pos=torch.from_numpy(vertices),
                face=torch.from_numpy(polygons.transpose()))

    # Attach the raw path for identification
    data.dir = path

    return data
