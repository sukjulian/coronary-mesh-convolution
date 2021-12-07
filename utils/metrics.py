import torch
import tqdm
from prettytable import PrettyTable


class Metrics(object):
    """Evaluation metrics.

    Args:
        loaders (list): list of data loaders containing all training samples.
    """

    def __init__(self, loaders):
        self.loaders = loaders

        self.scalar = None
        self.M = self.maximum_value  # avoid unnecessary loops through the loaders

    @property
    def maximum_value(self):

        # Maximum target value (across all samples) for normalisation of the L1 error
        maximum = torch.tensor(0.)
        for loader in self.loaders:
            for batch in tqdm.tqdm(loader):

                current = self.batch_maximum(batch)
                if current > maximum:
                    maximum = current

        return maximum

    def batch_maximum(self, batch):

        # Scalar or vector-valued labels
        if len(batch.y.shape) == 1:
            magnitude = torch.abs(batch.y)
            self.scalar = True
        else:
            magnitude = torch.linalg.norm(batch.y, dim=1)
            self.scalar = False

        return torch.max(magnitude)

    def approximation_error(self, prediction, reference):

        # Compare w.r.t. the magnitude of the difference vector (not element-wise)
        difference = reference - prediction
        if not self.scalar:
            difference = torch.linalg.norm(difference, dim=1)
            reference = torch.linalg.norm(reference, dim=1)

        # Approximation error from Su et al. (2020)
        return torch.sqrt(torch.sum(difference ** 2) / torch.sum(reference ** 2))

    def absolute_differences(self, prediction, reference):

        # Compare w.r.t. the magnitude of the difference vector (not element-wise)
        difference = reference - prediction
        if not self.scalar:
            difference = torch.linalg.norm(difference, dim=1)

        # Minimum, maximum and mean absolute difference
        delta = torch.abs(difference)
        minimum = torch.min(delta)
        maximum = torch.max(delta)
        mean = torch.mean(delta)  # mean absolute error

        return minimum, maximum, mean

    def statistics(self, model, device):

        # Accumulate the statistics over the data loader
        approximation_error = []
        normalised_mean_absolute_error = []
        delta_min, delta_max, delta_mean = [], [], []
        for loader in self.loaders:
            for batch in tqdm.tqdm(loader):

                batch = batch.to(device)
                prediction = model(batch)[batch.mask]
                label = batch.y[batch.mask]
                minimum, maximum, mean = self.absolute_differences(prediction, label)

                approximation_error.append(self.approximation_error(prediction, label).item())
                normalised_mean_absolute_error.append((mean / self.M).item())

                delta_min.append(minimum.item())
                delta_max.append(maximum.item())
                delta_mean.append(mean.item())

        # Statistical evaluation of the metrics
        table = PrettyTable(["Metric", "Mean", "Median", "75th percentile"])
        approximation_error = torch.tensor(approximation_error)
        table.add_row(["AE",
                       "{0:.1%}".format(torch.mean(approximation_error).item()),
                       "{0:.1%}".format(torch.median(approximation_error).item()),
                       "{0:.1%}".format(torch.quantile(approximation_error, 0.75).item())])
        normalised_mean_absolute_error = torch.tensor(normalised_mean_absolute_error)
        table.add_row(["NMAE",
                       "{0:.1%}".format(torch.mean(normalised_mean_absolute_error).item()),
                       "{0:.1%}".format(torch.median(normalised_mean_absolute_error).item()),
                       "{0:.1%}".format(torch.quantile(normalised_mean_absolute_error, 0.75).item())])
        delta_min, delta_max, delta_mean = torch.tensor(delta_min), torch.tensor(delta_max), torch.tensor(delta_mean)
        table.add_row(["D_min",
                       "{:.1f}".format(torch.mean(delta_min).item()),
                       "{:.1f}".format(torch.median(delta_min).item()),
                       "{:.1f}".format(torch.quantile(delta_min, 0.75).item())])
        table.add_row(["D_max",
                       "{:.1f}".format(torch.mean(delta_max).item()),
                       "{:.1f}".format(torch.median(delta_max).item()),
                       "{:.1f}".format(torch.quantile(delta_max, 0.75).item())])
        table.add_row(["D_mean",
                       "{:.1f}".format(torch.mean(delta_mean).item()),
                       "{:.1f}".format(torch.median(delta_mean).item()),
                       "{:.1f}".format(torch.quantile(delta_mean, 0.75).item())])

        return table
