import yaml
from types import SimpleNamespace


def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # allows config.model.backbone instead of config['model']['backbone']
    def dict_to_namespace(d):
        return SimpleNamespace(
            **{
                k: dict_to_namespace(v) if isinstance(v, dict) else v
                for k, v in d.items()
            }
        )

    return dict_to_namespace(config_dict)


def print_config(config):
    print(f"Model: {config.model.name}")
    print(f"Backbone: {config.model.backbone.name}")
    print(f"Optimizer: {config.training.optimizer.name}")
    print(f"Learning Rate: {config.training.optimizer.lr}")
    print(f"Scheduler: {config.training.scheduler.name}")
