import torch
from tqdm import tqdm
from prettytable import PrettyTable


class Metrics(object):
    """Evaluation metrics.

    Args:
        loaders (list): list of data loaders containing all test samples.
    """

    def __init__(self, loaders):
        self.loaders = loaders

        self.label_dimensions = None
        self.M = self.maximum_value  # avoid unnecessary loops through the loaders

    @property
    def maximum_value(self):

        # Maximum target value (across all samples) for normalisation of the L1 error
        maximum = torch.tensor(0.)
        for loader in self.loaders:
            for batch in loader:

                current = self.batch_maximum(batch)
                if current > maximum:
                    maximum = current

        return maximum

    def batch_maximum(self, batch):

        # Scalar or vector-valued labels
        self.label_dimensions = len(batch.y.shape)

        if self.label_dimensions == 1:
            magnitude = torch.abs(batch.y)
        elif self.label_dimensions == 2:
            magnitude = torch.linalg.norm(batch.y, dim=1)
        elif self.label_dimensions == 3:
            magnitude, _ = torch.median(torch.linalg.norm(batch.y, dim=2), dim=1)  # median over time

        return torch.max(magnitude)

    @staticmethod
    def cosine_similarity(prediction, reference):
        similarity = torch.nn.CosineSimilarity(dim=-1).forward(prediction, reference)

        return torch.min(similarity), torch.max(similarity), torch.mean(similarity)

    def approximation_error(self, prediction, reference):

        # Compare w.r.t. the magnitude of the difference vector (not element-wise)
        difference = reference - prediction

        if self.label_dimensions >= 2:
            difference = torch.linalg.norm(difference, dim=-1)
            reference = torch.linalg.norm(reference, dim=-1)

        # Approximation error from Su et al. (2020)
        return torch.sqrt(torch.sum(difference ** 2) / torch.sum(reference ** 2))

    def absolute_differences(self, prediction, reference):

        # Compare w.r.t. the magnitude of the difference vector (not element-wise)
        difference = reference - prediction

        if self.label_dimensions >= 2:
            difference = torch.linalg.norm(difference, dim=-1)

        # Minimum, maximum and mean absolute difference
        delta = torch.abs(difference)
        minimum = torch.min(delta)
        maximum = torch.max(delta)
        mean = torch.mean(delta)  # mean absolute error

        return minimum, maximum, mean, delta

    def scale(self, reference):

        if self.label_dimensions >= 2:
            reference = torch.linalg.norm(reference, dim=-1)

        return torch.max(reference), torch.median(reference)

    def statistics(self, model, device):

        # CPU should be faster
        model.to(device)

        # Accumulate the statistics over the data loader
        approximation_error = []
        normalised_mean_absolute_error = []
        delta_max, delta_mean = [], []
        # delta_full = []
        scale_max, scale_median = [], []
        cos_mean = []
        for loader in self.loaders:
            for batch in tqdm(loader):

                batch = batch.to(device)
                prediction = model(batch)[batch.mask]
                label = batch.y[batch.mask]
                _ , maximum, mean, delta = self.absolute_differences(prediction, label)

                approximation_error.append(self.approximation_error(prediction, label).item())
                normalised_mean_absolute_error.append((mean / self.M).item())

                delta_max.append(maximum.item())
                delta_mean.append(mean.item())

                # delta_full.append(delta)

                scale_max.append(self.scale(label)[0])
                scale_median.append(self.scale(label)[1])

                _, _, mean = self.cosine_similarity(prediction, label)
                cos_mean.append(mean.item())

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
        delta_max, delta_mean = torch.tensor(delta_max), torch.tensor(delta_mean)
        table.add_row(["D_max",
                       "{:.1f}".format(torch.mean(delta_max).item()),
                       "{:.1f}".format(torch.median(delta_max).item()),
                       "{:.1f}".format(torch.quantile(delta_max, 0.75).item())])
        table.add_row(["D_mean",
                       "{:.1f}".format(torch.mean(delta_mean).item()),
                       "{:.1f}".format(torch.median(delta_mean).item()),
                       "{:.1f}".format(torch.quantile(delta_mean, 0.75).item())])
        scale_max, scale_median = torch.tensor(scale_max), torch.tensor(scale_median)
        table.add_row(["L_max",
                       "{:.1f}".format(torch.mean(scale_max).item()),
                       "{:.1f}".format(torch.median(scale_max).item()),
                       "{:.1f}".format(torch.quantile(scale_max, 0.75).item())])
        table.add_row(["L_median",
                       "{:.1f}".format(torch.mean(scale_median).item()),
                       "{:.1f}".format(torch.median(scale_median).item()),
                       "{:.1f}".format(torch.quantile(scale_median, 0.75).item())])
        cos_mean = torch.tensor(cos_mean)
        table.add_row(["CS_mean",
                       "{:.2f}".format(torch.mean(cos_mean).item()),
                       "{:.2f}".format(torch.median(cos_mean).item()),
                       "{:.2f}".format(torch.quantile(cos_mean, 0.75).item())])

        return table
