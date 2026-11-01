import argparse
import logging
import os

from src.config import settings
from src.training.trainer import load_config, train

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train your Personal LLM on Open-Platypus"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=settings.CONFIG_PATH,
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test run with only 100 samples and 1 epoch",
    )
    args = parser.parse_args()
    if args.quick_test:
        import yaml

        config = load_config(args.config)
        config["dataset"]["max_samples"] = 100
        config["training"]["epochs"] = 1
        config["training"]["save_steps"] = 50
        config["output_dir"] = "./output/test-run"
        temp_config = "config/_temp_test_config.yaml"
        with open(temp_config, "w") as f:
            yaml.dump(config, f)
        logger.info("Quick test mode: 100 samples, 1 epoch")
        train(temp_config)
        os.remove(temp_config)
    else:
        train(args.config)


if __name__ == "__main__":
    main()
