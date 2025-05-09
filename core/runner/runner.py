import pdb

import hydra.utils

from core.system import *
import torch
import torch.distributed as dist
from core.tasks import tasks

from pytorch_lightning import Trainer

def set_seed(seed):
    pl.seed_everything(seed)

def set_device(device_config):
    # set the global cuda device
    torch.backends.cudnn.enabled = True
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
    torch.cuda.set_device(device_config.cuda)
    torch.set_float32_matmul_precision('high')
    # warnings.filterwarnings("always")


def set_processtitle(cfg):
    # set process title
    import setproctitle
    setproctitle.setproctitle(cfg.process_title)

def init_experiment(cfg, **kwargs):

    """print("config:")
    for k, v in cfg.items():
        print(k, v)
    print("=" * 20)

    print("kwargs:")
    for k, v in kwargs.items():
        print(k, v)
    print("=" * 20)"""

    # set seed
    set_seed(cfg.seed)

    # set process title
    set_processtitle(cfg)


def train_generation(cfg):
    init_experiment(cfg)
    system_cls = systems[cfg.system.name]
    system = system_cls(cfg)
    datamodule = system.get_task().get_param_data()
    # running
    trainer: Trainer = hydra.utils.instantiate(cfg.system.train.trainer)
    trainer.fit(system, datamodule=datamodule, ckpt_path=cfg.load_system_checkpoint)
    trainer.test(system, datamodule=datamodule)

    return {}

def test_generation(cfg):
    init_experiment(cfg)
    system_cls = systems[cfg.system.name]
    system = system_cls(cfg)
    datamodule = system.get_task().get_param_data()
    # running
    trainer: Trainer = hydra.utils.instantiate(cfg.system.train.trainer)
    trainer.test(system, datamodule=datamodule, ckpt_path=cfg.load_system_checkpoint)

    return {}

def train_task_for_data(cfg, **kwargs):
    init_experiment(cfg, **kwargs)
    task_cls = tasks[cfg.task.name]
    task = task_cls(cfg.task, **kwargs)

    task_result = task.train_for_data()
    return task_result
