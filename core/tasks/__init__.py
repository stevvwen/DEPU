from .classification import CFTask
from .rl_task import RLTask

tasks = {
    'classification': CFTask,
    'rl': RLTask,
}