import os


def experiment(model, dataset, batch_size, transform, epochs, lr, optimiser, objective):

    # Log all important settings to identify the experiment
    log = model.parameter_table().get_string() + "\n"
    log += "\nDataset: " + dataset
    log += "\nBatch size: " + str(batch_size) + "\n"
    log += "\nTransforms: " + str(transform) + "\n"
    log += "\nEpochs: " + str(epochs)
    log += "\nLearning rate: " + str(lr) + "\n"
    log += "\nOptimised using: " + str(optimiser)
    log += "\nLoss function: " + str(objective) + "\n"

    # Write to file in the visualisation folder (for convenience)
    if not os.path.exists('vis'):
        os.makedirs('vis')
    file = open("vis/log.txt", 'w')
    file.write(log)
    file.close()
