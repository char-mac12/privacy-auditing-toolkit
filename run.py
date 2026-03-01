import argparse
from pathlib import Path

from core.config_loader import ConfigLoader
from core.attack_runner import AttackRunner

import models
import dataset_loaders
import attacks
import metrics
import reports

import attacks.loss_based_mia    # this was required but shouldn't be
import attacks.min_k_mia         # this was required but shouldn't be


def main():
    parser = argparse.ArgumentParser(description="Run privacy attacks")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pythia_160m_pile_cc.json",
        help="Path to config JSON file"
    )
    args = parser.parse_args()
    
    config = ConfigLoader.load(args.config)
    AttackRunner(config).run()


if __name__ == "__main__":
    main()