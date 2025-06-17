import sys, os
USE_WANDB = False
import torch.nn.functional as F
# set global seed
import random
import numpy as np
import torch
seed = SEED = 430
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# other
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
if USE_WANDB: import wandb
import torch.optim as optim
# model
from core.pdiff.pdiff import PDiff as Model
from core.pdiff.pdiff import OneDimVAE as VAE
from core.pdiff.diffusion import DDPMSampler, DDIMSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate.utils import DistributedDataParallelKwargs, AutocastKwargs
from accelerate import Accelerator
# dataset
from torch.utils.data import DataLoader, TensorDataset

config = {
    "seed": SEED,
    # dataset setting
    "dataset": "Customized", #TODO: change this to your dataset name
    "sequence_length": 'auto',
    # train setting
    "batch_size": 50,
    "num_workers": 4,
    "total_steps": 1000,  # diffusion training steps
    "vae_steps": 500,  # vae training steps
    "learning_rate": 0.0001,  # diffusion learning rate
    "vae_learning_rate": 0.00002,  # vae learning rate
    "weight_decay": 0.0,
    "save_every": 500,
    "print_every": 20,
    "autocast": lambda i: True,
    "checkpoint_save_path": "./checkpoint",
    # test setting
    "test_batch_size": 1,  # fixed, don't change this
    #"generated_path": Dataset.generated_path,
    #"test_command": Dataset.test_command, #TODO: change this to your test command
    # to log
    "model_config": {
        # diffusion config
        "layer_channels": [1, 64, 128, 256, 512, 256, 128, 64, 1],  # channels of 1D CNN
        "model_dim": 128,  # latent dim of vae
        "kernel_size": 7,
        "sample_mode": DDPMSampler,
        "beta": (0.0001, 0.02),
        "T": 1000,
        # vae config
        "channels": [64, 128, 256, 256, 32],
    },
    "tag": os.path.basename(__file__)[:-3],
}




data = torch.load("param_data/Pendulum-v1_current_policy/data.pt")
pdata = data['pdata']

print("Dataset length:", len(pdata))
print("input shape:", pdata[0].shape)

# Pad each tensor in pdata if its length isn't a multiple of 64
divide_slice_length = 64
current_length = pdata[0].shape[0]
if current_length % divide_slice_length != 0:
    pad_amount = divide_slice_length - (current_length % divide_slice_length)
    print(f"Padding input from {current_length} to {current_length + pad_amount}")
    pdata = F.pad(pdata, (0, pad_amount), mode='constant', value=0)

if config["sequence_length"] == "auto":
    config["sequence_length"] = pdata[0].shape[0]
    print(f"sequence length: {config['sequence_length']}")

train_set= TensorDataset(pdata)

train_loader = DataLoader(
    dataset=train_set,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    persistent_workers=True,
    drop_last=True,
    shuffle=True,
)

# Model
print('==> Building model..')
Model.config = config["model_config"]
model = Model(sequence_length=config["sequence_length"])
vae = VAE(d_model=config["model_config"]["channels"],
          d_latent=config["model_config"]["model_dim"],
          sequence_length=config["sequence_length"],
          kernel_size=config["model_config"]["kernel_size"])

# Optimizer
print('==> Building optimizer..')
vae_optimizer = optim.AdamW(
    params=vae.parameters(),
    lr=config["vae_learning_rate"],
    weight_decay=config["weight_decay"],
)
optimizer = optim.AdamW(
    params=model.parameters(),
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
)
vae_scheduler = CosineAnnealingLR(
    optimizer=vae_optimizer,
    T_max=config["vae_steps"],
)
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=config["total_steps"],
)

# accelerator
if __name__ == "__main__":
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs,])
    vae, model, vae_optimizer, optimizer, train_loader = \
            accelerator.prepare(vae, model, vae_optimizer, optimizer, train_loader)

# wandb
if __name__ == "__main__" and USE_WANDB and accelerator.is_main_process:
    wandb.login(key="your_api_key")
    wandb.init(project="AR-Param-Generation", name=config['tag'], config=config,)

# Training
print('==> Defining training..')

def train_vae():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> start training vae..")
    vae.train()
    for batch_idx, (param,) in enumerate(train_loader):
        vae_optimizer.zero_grad()
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            param = param.flatten(start_dim=1)
            param += torch.randn_like(param) * 0.001
            loss = vae(x=param, use_var=True, manual_std=0.1, kld_weight=0.0)
        accelerator.backward(loss)
        vae_optimizer.step()
        if accelerator.is_main_process:
            vae_scheduler.step()
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"vae_loss": loss.item()})
        elif USE_WANDB:
            pass
        else:
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx >= config["vae_steps"]:
            break

def train():
    if not USE_WANDB:
        train_loss = 0
        this_steps = 0
    print("==> start training..")
    model.train()
    for batch_idx, (param,) in enumerate(train_loader):
        optimizer.zero_grad()
        with accelerator.autocast(autocast_handler=AutocastKwargs(enabled=config["autocast"](batch_idx))):
            param = param.flatten(start_dim=1)
            with torch.no_grad():
                mu, _ = vae.encode(param)
            loss = model(x=mu)
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            scheduler.step()
        if USE_WANDB and accelerator.is_main_process:
            wandb.log({"train_loss": loss.item()})
        elif USE_WANDB:
            pass
        else:
            train_loss += loss.item()
            this_steps += 1
            if this_steps % config["print_every"] == 0:
                print('Loss: %.6f' % (train_loss/this_steps))
                this_steps = 0
                train_loss = 0
        if batch_idx % config["save_every"] == 0 and accelerator.is_main_process:
            os.makedirs(config["checkpoint_save_path"], exist_ok=True)
            state = {"diffusion": accelerator.unwrap_model(model).state_dict(), "vae": vae.state_dict()}
            torch.save(state, os.path.join(config["checkpoint_save_path"], f"{config['tag']}.pth"))
            #generate(save_path=config["generated_path"], need_test=True)
        if batch_idx >= config["total_steps"]:
            break

"""def generate(save_path=config["generated_path"], need_test=True):
    print("\n==> Generating..")
    model.eval()
    with torch.no_grad():
        mu = model(sample=True)
        prediction = vae.decode(mu)
        generated_norm = prediction.abs().mean()
    print("Generated_norm:", generated_norm.item())
    if USE_WANDB:
        wandb.log({"generated_norm": generated_norm.item()})
    prediction = prediction.view(-1, divide_slice_length)
    train_set.save_params(prediction, save_path=save_path)
    if need_test:
        os.system(config["test_command"])
        print("\n")
    model.train()
    return prediction"""

if __name__ == '__main__':
    train_vae()
    vae = accelerator.unwrap_model(vae)
    train()
    del train_loader
    print("Finished Training!")
    exit(0)
