import torch_geometric
import vtk
from utils import dataset_tools
import torch
import os
from pathlib import Path
import tqdm


# Dataset consisting of VTP files (exceeding available memory)
class LargeVesselDataset(torch_geometric.data.Dataset):
    def __init__(self, root, pattern, split, purpose, transform=None, pre_transform=None):

        # Loading functionality
        self.root = root
        self.pattern = pattern
        self.purpose = purpose
        self.reader = vtk.vtkXMLPolyDataReader()  # avoid creating one object per sample

        # Training and validation split
        self.split = split

        super(LargeVesselDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return dataset_tools.raw_file_names(self)

    @property
    def processed_file_names(self):
        root = os.path.join(self.root, "processed", self.purpose)
        Path(root).mkdir(parents=True, exist_ok=True)
        return [os.path.join(root, "data_{}.pt".format(i)) for i in range(len(self.raw_paths))]

    def download(self):
        raise RuntimeError("Dataset not found.")

    def process(self):
        i = 0
        for path in tqdm.tqdm(self.raw_paths):
            data = dataset_tools.load_process(self, path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Save to disk
            torch.save(data, os.path.join(self.processed_dir, self.purpose, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.purpose, 'data_{}.pt'.format(idx)))
        return data
