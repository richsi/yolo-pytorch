from models import MODEL
import argparse
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/yolov1_pascal.yaml)",
    )

    parser.add_argument(
        "--device",
        type = str,
        default = "gpu",
        choices=["cpu", "gpu", "mps"],
        help="Device to train on"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    model = MODEL[config.model.name]
    



if __name__ == "__main__":
    main()
