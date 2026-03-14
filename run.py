import argparse
from pathlib import Path

from core.config_loader import ConfigLoader
from core.audit_runner import AuditRunner

import model_interfaces
import dataset_loaders
import attack_executor
import metrics_calculator
import report_generator

import attack_executor.loss_based_mia    # this was required but shouldn't be
import attack_executor.min_k_mia         # this was required but shouldn't be
import attack_executor.range_mia


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
    AuditRunner(config).run()


if __name__ == "__main__":
    main()