import torch_geometric
from datasets import InMemoryVesselDataset
import os
from utils.model_tools import load
from torch_geometric.nn.data_parallel import DataParallel
import torch
from utils import visualisation
from utils.metrics import Metrics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm


dataset_folder = "vessel-datasets"


class Experiment:
    def __init__(self, model, dataset, batch_size, tag, transforms=None, lr=0.001, epochs=1000, parallel=None):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.tag = tag

        # Data transforms
        transforms_list = []  # dummy list
        if transforms is not None:
            transforms_list.extend(transforms)
        self.transform = torch_geometric.transforms.Compose(transforms_list)

        # Optional parameters
        self.lr = lr
        self.epochs = epochs
        self.parallel = parallel

    def run(self, device):

        if self.dataset == 'bifurcating':
            self.fit_bifurcating(device)

        elif self.dataset == 'single':
            self.fit_single(device)

        else:
            raise RuntimeError("Dataset not found.")

    def fit_bifurcating(self, device):

        # Dataset IDs
        path = f"{dataset_folder}/stead/bifurcating/"
        pattern = "*"  # in the "raw" folder

        # Training, validation and test split (total 1999 samples)
        train_split = [0, 1600]
        valid_split = [1600, 1799]
        test_split = [1799, 1999]

        # Data loader parameters
        params = {'batch_size': self.batch_size, 'num_workers': 0, 'pin_memory': False, 'shuffle': True}

        args = self.tag, path, pattern, train_split, valid_split, test_split, params
        self.fit(device, *args)

    def fit_single(self, device):

        # Dataset IDs
        path = f"{dataset_folder}/stead/single/"  # directory containing the folder "raw"
        pattern = "*"  # in the "raw" folder

        # Training, validation and test split (total 2000 samples)
        train_split = [0, 1600]
        valid_split = [1600, 1800]
        test_split = [1800, 2000]

        # Data loader parameters
        params = {'batch_size': self.batch_size, 'num_workers': 0, 'pin_memory': False, 'shuffle': True}

        args = self.tag, path, pattern, train_split, valid_split, test_split, params
        self.fit(device, *args)

    def fit(self, device, tag, path, pattern, train_split, valid_split, test_split, params):

        # Create datasets
        train = InMemoryVesselDataset(path, pattern, train_split, "train", pre_transform=self.transform)
        valid = InMemoryVesselDataset(path, pattern, valid_split, "valid", pre_transform=self.transform)
        test = InMemoryVesselDataset(path, pattern, test_split, "test", pre_transform=self.transform)

        # Data loaders
        if self.parallel is not None:
            train_loader = torch_geometric.loader.DataListLoader(train, **params)
            valid_loader = torch_geometric.loader.DataListLoader(valid, **params)
        else:
            train_loader = torch_geometric.loader.DataLoader(train, **params)
            valid_loader = torch_geometric.loader.DataLoader(valid, **params)
        test_loader = torch_geometric.loader.DataLoader(test, batch_size=1)

        # Network model
        model = self.model.to(device)
        if os.path.exists("model-weights/{}.pt".format(tag)):
            print("Resuming from pre-trained configuration '{}.pt'".format(tag))
            model.load_state_dict(load("model-weights/{}.pt".format(tag), map_location=device))
        if self.parallel is not None:
            model = DataParallel(model, device_ids=self.parallel)

        # Optimisation settings
        objective = torch.nn.L1Loss()
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Training
        args = model, [train_loader, valid_loader], objective, self.epochs, optimiser, device, tag
        if self.parallel is not None:
            parallel_training(*args)
        else:
            training(*args)

        # Write predictions to VTP files for visualisation
        model = self.model  # change from parallel to serial
        model.load_state_dict(load("model-weights/" + tag + ".pt", map_location=device))
        model.eval()  # set to evaluation mode
        if not os.path.exists('vis'):
            os.makedirs('vis')
        with torch.no_grad():
            for i, sample in enumerate(tqdm.tqdm(test_loader, desc="Visualising")):
                prediction = model.to(torch.device('cpu'))(sample.to(torch.device('cpu')))
                target = sample.y.clone()

                fields = visualisation.default_fields(sample, prediction)
                filename = "vis/prediction{:03d}.vtp".format(i)
                visualisation.new_file(
                    sample.pos,
                    sample.face if hasattr(sample, 'face') else None,
                    filename,
                    fields
                )

            # Tabulate the evaluation metrics
            table = Metrics([test_loader]).statistics(model, torch.device('cpu'))
            print(table)


def training(model, loaders, objective, epochs, optimiser, device, tag, use_scheduler=False):

    # Data loaders
    train_loader, valid_loader = loaders

    # Configure TensorBoard
    writer = SummaryWriter()  # $ tensorboard --logdir=runs --port=6006

    # Hack$
    best = 1e8

    for epoch in tqdm.tqdm(range(epochs), desc="Training"):
        model.train()  # set to training mode
        train_loss = torch.zeros(len(train_loader))
        for i, batch in enumerate(train_loader):

            optimiser.zero_grad()  # reset the gradient accumulator
            batch = batch.to(device)  # transfer to GPU if available

            output = model(batch)

            labels = batch.y
            loss = objective(output[batch.mask], labels[batch.mask])  # masked optimisation
            train_loss[i] = loss.detach().clone()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimiser.step()

        writer.add_scalar("Loss/train", torch.median(train_loss), epoch)
        model.eval()  # set to evaluation mode

        valid_loss = torch.zeros(len(valid_loader))
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                batch = batch.to(device)  # transfer to GPU if available
                output = model(batch)[batch.mask]
                labels = batch.y[batch.mask]
                loss = objective(output, labels)
                valid_loss[i] = loss.detach().clone()

        writer.add_scalar("Loss/validate", torch.median(valid_loss), epoch)

        # Hack$
        if torch.median(valid_loss) < best:
            if not os.path.exists('model-weights'):
                os.makedirs('model-weights')
            torch.save(model.state_dict(), "model-weights/{}_best_validation.pt".format(tag))
            best = torch.median(valid_loss)

    # Save the model for inference
    if not os.path.exists('model-weights'):
        os.makedirs('model-weights')
    torch.save(model.state_dict(), "model-weights/" + tag + ".pt")


def parallel_training(model, loaders, objective, epochs, optimiser, device, tag):

    # Data loaders
    train_loader, valid_loader = loaders

    # Configure TensorBoard
    writer = SummaryWriter()  # $ tensorboard --logdir=runs --port=6006

    # Hack$
    best = 1e8

    for epoch in tqdm.tqdm(range(epochs), desc="Training"):
        model.train()  # set to training mode
        train_loss = torch.zeros(len(train_loader))
        for i, batch in enumerate(train_loader):

            optimiser.zero_grad()  # reset the gradient accumulator
            batch = [sample.to(device) for sample in batch]  # transfer to GPU if available

            output = model(batch)

            mask = torch.cat([sample.mask for sample in batch], dim=0)
            labels = torch.cat([sample.y for sample in batch], dim=0)
            loss = objective(output[mask], labels[mask])  # masked optimisation
            train_loss[i] = loss.detach().clone()

            loss.backward()
            optimiser.step()

        writer.add_scalar("Loss/train", torch.median(train_loss), epoch)
        model.eval()  # set to evaluation mode

        valid_loss = torch.zeros(len(valid_loader))
        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                batch = [sample.to(device) for sample in batch]  # transfer to GPU if available
                labels = torch.cat([sample.y for sample in batch], dim=0)
                mask = torch.cat([sample.mask for sample in batch], dim=0)
                output = model(batch)
                loss = objective(output[mask], labels[mask])
                valid_loss[i] = loss.detach().clone()

        writer.add_scalar("Loss/validate", torch.median(valid_loss), epoch)

        # Hack$
        if torch.median(valid_loss) < best:
            if not os.path.exists('model-weights'):
                os.makedirs('model-weights')
            torch.save(model.state_dict(), "model-weights/{}_best_validation.pt".format(tag))
            best = torch.median(valid_loss)

    # Save the model for inference
    if not os.path.exists('model-weights'):
        os.makedirs('model-weights')
    torch.save(model.state_dict(), "model-weights/" + tag + ".pt")
