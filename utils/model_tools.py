from collections import OrderedDict
import torch

import os

from prettytable import PrettyTable


# Get rid of the prefix "module." in the state dict
def parallel_to_serial(ordered_dict):
    return OrderedDict((key[7:], value) for key, value in ordered_dict.items())


# Return ordered state dictionary for serial data model
def load(path, map_location):
    ordered_dict = torch.load(path, map_location)
    if next(iter(ordered_dict)).startswith("module."):
        return parallel_to_serial(ordered_dict)
    else:
        return ordered_dict


def parameter_table(model):
    table = PrettyTable(["Modules", "Parameters"])
    total = 0

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total += params
    table.add_row(["TOTAL", total])

    return table
