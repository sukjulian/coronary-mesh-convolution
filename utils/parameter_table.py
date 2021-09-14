from prettytable import PrettyTable


def create(model):
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
