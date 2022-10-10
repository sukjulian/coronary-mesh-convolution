import torch_geometric
import vtk
import torch
import os
from pathlib import Path
import tqdm
import glob
import h5py
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from data import MultiscaleData as Data
import utils.vtk_tools


# Vessel dataset fitting in RAM
class InMemoryVesselDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, pattern, split, purpose, transform=None, pre_transform=None):

        # Loading functionality
        self.root = root
        self.pattern = pattern
        self.purpose = purpose
        self.reader = vtk.vtkXMLPolyDataReader()

        # Training and validation split
        self.split = split

        super(InMemoryVesselDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        root = os.path.join(self.root, "raw")
        absolute = glob.glob(os.path.join(root, self.pattern))
        absolute.sort()  # comparability
        if ".hdf5" in os.path.basename(absolute[0]):
            with h5py.File(absolute[0], 'r') as f:
                absolute = [os.path.join(absolute[0], i) for i in list(f)]
        absolute = absolute[self.split[0]:self.split[1]]

        return [os.path.relpath(a, root) for a in absolute]

    @property
    def processed_file_names(self):
        root = os.path.join(self.root, "processed", self.purpose)
        Path(root).mkdir(parents=True, exist_ok=True)
        return [os.path.join(self.purpose, "data.pt")]

    def download(self):
        return

    def process(self):
        data_list = []
        for path in tqdm.tqdm(self.raw_paths):
            data = self.load_process_hdf5(path)

            # Append to data list
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            transformed = []
            for data in tqdm.tqdm(data_list):
                transformed.append(self.pre_transform(data))
            data_list = transformed

        # Save to disk
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @staticmethod
    def load_process_hdf5(path):
        file, sample = os.path.split(path)

        with h5py.File(file, 'r') as f:

            # Load into PyTorch geometric
            data = Data(
                shape_id=torch.tensor(f[sample].attrs['shape id']) if 'shape id' in f[sample].attrs else None,
                condition=torch.tensor(f[sample]['cbf'][()]) if 'cbf' in f[sample] else None,
                t=torch.from_numpy(f[sample]['t'][()][None, ...]) if 't' in f[sample] else None,  # expand dimension for batching
                y=torch.from_numpy(f[sample]['wss'][()].swapaxes(0, -2)),  # swap axes for correct batching
                inlet_index=torch.from_numpy(f[sample]['inlet_idx' if 'inlet_idx' in f[sample] else 'inlet_idcs'][()]),
                pos=torch.from_numpy(f[sample]['pos'][()]),
                face=torch.from_numpy(f[sample]['face'][()].T).long()  # transpose to match PyG convention
            )

        return data
