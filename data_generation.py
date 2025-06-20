import hydra
from omegaconf import DictConfig
from utils import init_experiment
from data import RLTask


@hydra.main(config_path="configs", config_name="base", version_base='1.2')
def training_for_data(config: DictConfig):
    init_experiment(config)
    task= RLTask(config)
    result = task.train_for_data()
    print("the data save at {}".format(result['save_path']))
    return


if __name__ == "__main__":
    training_for_data()