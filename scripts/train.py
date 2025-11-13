from models import MODEL
import argparse
from utils.config import load_config, print_config


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

    print(f"Using config: {args.config}")

    config = load_config(args.config)

    print_config(config)




if __name__ == "__main__":
    main()
