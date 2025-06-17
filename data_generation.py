import hydra
from omegaconf import DictConfig
from core.utils.utils import init_experiment
from core.tasks import tasks


def train_task_for_data(cfg, **kwargs):
    init_experiment(cfg, **kwargs)
    task_cls = tasks[cfg.task.name]
    task = task_cls(cfg.task, **kwargs)

    task_result = task.train_for_data()
    return task_result


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_for_data(config: DictConfig):
    result = train_task_for_data(config)
    print("the data save at {}".format(result['save_path']))
    return


if __name__ == "__main__":
    training_for_data()