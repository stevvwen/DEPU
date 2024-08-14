from .classification import CFTask
from .rl import RLTask

tasks = {
    'classification': CFTask,
    'rl': RLTask,
}