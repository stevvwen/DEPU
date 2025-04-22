import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from core.module.modules.encoder import medium
from core.module.modules.encoder import medium

# === Load data ===
param_path = "param_data/Pendulum-v1/data.pt"
param_dict = torch.load(param_path)
param_data = param_dict['pdata']  # Should be a tensor of shape [N, D]
if isinstance(param_data, list):
    param_data = torch.stack(param_data)  # Convert list of tensors to a single tensor



# normalize the data
mean= param_data.mean(dim=0)
std= param_data.std(dim=0)
param_data = (param_data - mean) / std

# === Create dataset and dataloader ===
batch_size = 64
dataset = TensorDataset(param_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === Define model, optimizer, and loss ===
input_dim = param_data.shape[1]
#ae = medium(input_dim, 0.05, 0.1)
ae = medium(input_dim, 0.05, 0.3)
optimizer = optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# === Training loop ===
num_epochs = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae = ae.to(device)

for epoch in range(num_epochs):
    ae.train()
    total_loss = 0.0

    for batch in dataloader:
        x = batch[0].to(device)

        # Forward pass
        x_recon = ae(x)
        loss = loss_fn(x_recon, x)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(param_data)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

