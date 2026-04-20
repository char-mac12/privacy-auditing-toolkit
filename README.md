# Privacy Auditing Toolkit for Large Language Models
This toolkit provides a framework for auditing privacy leakage in large language models (LLMs). It is modular to ensure extensibility and audits are run through JSON configuration files to ensure reproducibility. 

## Features
### Attacks
Membership inference attacks (MIAs) are supported using a registry-based interface. Each attack inherits from an abstract base class ensuring that all attacks:
* Compute membership scores for member and non-member examples
* Return a structured ```AttackResult``` object for evaluation and reporting

This allows new attacks to be added easily, the steps are:
* Create a new file in the ```attack_executor``` folder that inherits from the abstract base class ```base_attack```
* Implement the required attributes and methods
* Register with an ID using the registry pattern

There are three supported attack types:
* ```loss-based-mia```: Uses the model's sequence loss as a membership signal. Lower loss implies a sample was more likely to be used in training.
* ```min-k-mia```: Computes the average lowest-probability tokens in a sequence.
* ```range-mia```: Generates a neighbourhood of altered samples using masked word replacement with a masked language model. A base attack is executed on the neighbourhood and scores are aggregated using a trimmed average.

### Model interfaces
Multiple model families and types can be supported using a registry-based interface. Currently, all models inherit from a shared ```CausalLmModel``` base class, which provides common functionality:
* Sequence loss computation
* Per-token loss calculation
* Text generation

Supported model types include:
* ```pythia```: Supports models from the EleutherAI Pythia suite
* ```gpt2```: Supports GPT-2 models
* ```llama```: Supports the LLaMA models

These models can be loaded from either hugging face model IDs or local file paths to downloaded models.

### Dataset loaders
Multiple dataset formats are supported using a registry-based interface. Supported dataset types include:
* ```saved-split```: Loads local pre-split member and non-member JSON files.
* ```pile```: Loads local datasets that are subsets of The Pile with predefined member and non-member splits.
* ```hugging-face```: Loads datasets directly from the HuggingFace Hub. An optional ```split``` can be specified.
* ```wikimia```: Loads the WIKIMIA benchmark dataset and separates samples based on their membership labels.
* ```wikitext```: Loads the WikiText dataset from the HuggingFace Hub. 

### Metrics calculation
Membership inference attacks are evaluated using the following metrics:
* AUC-ROC
* TPR@1%FPR
* TPR@0.1%FPR
* TPR@0.001%FPR
* Advantage (TPR - FPR at the optimal threshold)
* Accuracy
* Precision
* Recall
* F1-score

### Report generation
There are four types of report supported:
* Console
* CSV
* JSON
* PDF

## Project Structure
```text
privacy-auditing-toolkit/
│
├── attack_executor/        # Runs membership inference attacks
├── configs/                # JSON experiment configurations
├── core/                   # Shared core utilities including audit orchestration
├── dataset_loaders/        # Loads datasets
├── metrics_calculator/     # Computes evaluation metrics
├── model_interfaces/       # Model interfaces
├── report_generator/       # Generates experiment reports in different formats
├── reports/                # Saved outputs and results
├── scripts/                # Useful scripts
├── requirements.txt        # Dependencies
├── run.py                  # Main entry point
└── setup.py                # Package setup
```

## Installation
```bash
# Clone repository
git clone https://github.com/char-mac12/privacy-auditing-toolkit
cd privacy-auditing-toolkit

# Create environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration
All experiments are defined using JSON configuration files. If not specified, the toolkit will look for a config file in 
```/configs/privacy-audit.json```. 

However, a custom configuration file can be specified using a command line argument:
```bash
python run.py --config configs/your_config_file_name.json
```

### Example Minimal Configuration

```json
{
  "model": {
  "id": "pythia",
  "model_id": "models/your_pythia_model",
  "device": "cuda"
  }
  "dataset": {
    "id": "saved-split",
    "data_dir": "data/your_data_split",
    "member_file": "members.json",
    "non_member_file": "non_members.json"
  },
  "attack": {
    "id": "loss-based-mia"
  },
  "reporter": {
    "id": "pdf"
  }
}
```

### Reproducibility
* All experiments are defined using JSON configuration files
* Fixed random seeds are used where applicable

## Limitations
* Only three attacks have been implemented
* Dataset usage and model interface support is limited
* Limited fine-tuning and no privacy-preserving techniques like differential privacy are built in

## Future Work
* Implementation of more attack types
* Support for more datasets and model types
* Expanded fine-tuning support
* Differential privacy support
* More report types like HTML or a GUI

## Acknowledgement
This project was developed as part of a dissertation at the University of York on privacy leakage in large language models. It builds on prior work in membership inference attacks, PEFT (LoRA), the Pythia suite of models, ReportLab for PDF generation, MIMIR and The Pile.
