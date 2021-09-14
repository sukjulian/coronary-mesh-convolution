import torch_geometric
import vtk
import torch
from utils import dataset_tools
import os
from pathlib import Path
import tqdm


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
        return dataset_tools.raw_file_names(self)

    @property
    def processed_file_names(self):
        root = os.path.join(self.root, "processed", self.purpose)
        Path(root).mkdir(parents=True, exist_ok=True)
        return [os.path.join(root, "data.pt")]

    def download(self):
        raise RuntimeError("Dataset not found.")

    def process(self):
        data_list = []
        for path in tqdm.tqdm(self.raw_paths):
            data = dataset_tools.load_process(self, path)

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
