import os 
import ray 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ray.tune.schedulers import ASHAScheduler
from ray import train, tune
from ChebyKANLayer import ChebyKANLayer

# Define target function
def target_function(x):
    y = np.zeros_like(x)
    mask1 = x < 0.5
    y[mask1] = np.sin(20 * np.pi * x[mask1]) + x[mask1] ** 2
    mask2 = (0.5 <= x) & (x < 1.5)
    y[mask2] = 0.5 * x[mask2] * np.exp(-x[mask2]) + np.abs(np.sin(5 * np.pi * x[mask2]))
    mask3 = x >= 1.5
    y[mask3] = np.log(x[mask3] - 1) / np.log(2) - np.cos(2 * np.pi * x[mask3])
    return y

class ChebyKAN(nn.Module):
    def __init__(self, num_layers, hidden_dim, degree):
        super(ChebyKAN, self).__init__()
        layers = []
        input_dim = 1
        for _ in range(num_layers):
            layers.append(ChebyKANLayer(input_dim, hidden_dim, degree))
            input_dim = hidden_dim
        layers.append(ChebyKANLayer(hidden_dim, 1, degree))  # Output layer
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
def train_chebykan(config):
    # Data preparation
    x_train = torch.linspace(0, 2, steps=500).unsqueeze(1)
    y_train = torch.tensor(target_function(x_train.numpy()), dtype=torch.float32)
    
    x_val = x_train + 0.002
    y_val = torch.tensor(target_function(x_val.numpy()), dtype=torch.float32)

    # Model
    model = ChebyKAN(config["num_layers"], config["hidden_dim"], config["degree"])

    # Depending on the scaling method
    if config["scaling"] == "centralize":
        x_train -= 1
        x_val -= 1
    elif config["scaling"] == "minmax":
        min_x = torch.min(x_train)
        max_x = torch.max(x_train)
        x_train = (x_train - min_x) / (max_x - min_x)
        x_val = (x_val - min_x) / (max_x - min_x)
    elif config["scaling"] == "standard":
        mean_x = torch.mean(x_train)
        std_x = torch.std(x_train)
        x_train = (x_train - mean_x) / std_x
        x_val = (x_val - mean_x) / std_x

    # Loss and optimizer
    epochs = 200000
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):  # Use a smaller number of epochs for demonstration
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        sched.step()
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
        model.train()

        train.report({"loss": loss.item(), "val_loss": val_loss.item()})


# Hyperparameter space
config = {
    "lr": tune.loguniform(1e-4, 1e-2),
    "degree": tune.choice([2, 3, 4, 5, 6]),
    "hidden_dim": tune.choice([8, 16, 32, 64]),
    "num_layers": tune.choice([1, 2, 3, 4]),
    "scaling": tune.choice(["centralize", "minmax", "standard"])
}

# Scheduler and search algorithm
scheduler = ASHAScheduler(
    max_t=200000,
    grace_period=5000,
    reduction_factor=2)

local_dir_path = "C:/tune"
local_temp = "C:/tune_temp"
os.makedirs(local_dir_path, exist_ok=True)
os.makedirs(local_temp, exist_ok=True)

ray.init(_temp_dir=local_temp)
# Start the tuning
import time 
def trial_str_creator(trial):
    trialname = time.strftime("%Y-%m-%d_%H-%M-%S") + trial.trial_id
    return trialname


analysis = tune.run(
    train_chebykan,
    config=config,
    num_samples=10,
    max_concurrent_trials=8,
    scheduler=scheduler,
    local_dir=local_dir_path,
    metric="loss",
    mode="min",
    trial_name_creator=trial_str_creator,
    progress_reporter=tune.CLIReporter(metric_columns=["loss", "training_iteration"]))

print("Best hyperparameters found were: ", analysis.best_config)