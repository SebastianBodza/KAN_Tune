import os 
import ray 
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ray.tune.schedulers import ASHAScheduler
from ray import train, tune

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

# Define MLP
class SimpleMLP(nn.Module):
    def __init__(self, num_layers, num_hidden):
        super(SimpleMLP, self).__init__()
        layers = [nn.Linear(1, num_hidden), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(num_hidden, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def train_mlp(config):
    # Data preparation
    x_train = torch.linspace(0, 2, steps=500).unsqueeze(1)
    y_train = torch.tensor(target_function(x_train.numpy()), dtype=torch.float32)
    
    x_val = x_train + 0.002
    y_val = torch.tensor(target_function(x_val.numpy()), dtype=torch.float32)

    # Model
    model = SimpleMLP(config["num_layers"], config["num_hidden"])

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
    "num_hidden": tune.choice([32, 64, 128, 256]),
    "num_layers": tune.choice([1, 2, 3,4,5,6]),
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
    train_mlp,
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