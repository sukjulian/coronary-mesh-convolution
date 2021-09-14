from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch
import os


def fit(model, loaders, objective, epochs, optimiser, device, tag):

    # Data loaders
    train_loader, valid_loader = loaders

    # Configure TensorBoard
    writer = SummaryWriter()  # $ tensorboard --logdir=runs --port=6006

    for epoch in tqdm.tqdm(range(epochs)):
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
            optimiser.step()

        train_limits = {'mean': torch.mean(train_loss),
                        'median': torch.median(train_loss),
                        '75th': torch.quantile(train_loss, 0.75)}
        writer.add_scalars("Loss/train", train_limits, epoch)
        model.eval()  # set to evaluation mode

        valid_loss = torch.zeros(len(valid_loader))
        for i, batch in enumerate(valid_loader):
            batch = batch.to(device)  # transfer to GPU if available
            output = model(batch)[batch.mask]
            labels = batch.y[batch.mask]
            loss = objective(output, labels)
            valid_loss[i] = loss.detach().clone()

        valid_stats = {'mean': torch.mean(valid_loss),
                       'median': torch.median(valid_loss),
                       '75th': torch.quantile(valid_loss, 0.75)}
        writer.add_scalars("Loss/validate", valid_stats, epoch)

    # Save the model for inference
    if not os.path.exists('data'):
        os.makedirs('data')
    torch.save(model.state_dict(), "data/" + tag + ".pt")
