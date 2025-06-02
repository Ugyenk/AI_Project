import json
from torch.utils.tensorboard import SummaryWriter

# Load the results JSON
with open("hyperparameter_tuning_results.json", "r") as f:
    results = json.load(f)

# Create a SummaryWriter per model/hyperparam combination
for model_name, configs in results.items():
    for config_name, history in configs.items():
        log_dir = f"runs/{model_name}/{config_name}"
        writer = SummaryWriter(log_dir=log_dir)

        losses = history["loss"]
        accuracies = history["accuracy"]

        for epoch, (loss, acc) in enumerate(zip(losses, accuracies), start=1):
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Accuracy/train", acc, epoch)

        writer.close()

print("TensorBoard logs created in ./runs/")
print("Run `tensorboard --logdir=runs` to visualize.")
