"""
LoRA Fine-tuning for Pythia using the Hugging Face PEFT framework and a local dataset

Example usage:
    python lora.py \
        --model EleutherAI/pythia-160m-deduped \
        --data /local_datasets/mimir_pile_cc \
        --samples 100 \
        --epochs 4 \
        --output ./checkpoints/pythia_160m_lora
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

class NaNDetectionCallback(TrainerCallback):
    """Stop training if NaN/Inf loss detected"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss")
            loss_is_nan = torch.isnan(torch.tensor(loss))
            loss_is_inf = torch.isinf(torch.tensor(loss))
            if loss is not None and (loss_is_nan or loss_is_inf):
                print(f"\n ERROR: loss was NaN/Inf at step {state.global_step}")
                print(f"Loss value: {loss}")
                print(f"Stopping training")
                control.should_training_stop = True
        return control

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Pythia")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-1.4b-deduped",
                       help="Base model name from HuggingFace")
    
    # Dataset arguments
    parser.add_argument("--data", type=str, required=True,
                       help="Path to local dataset")
    parser.add_argument("--samples", type=int, default=10000,
                       help="Number of training samples")
    
    # LoRA hyperparameters
    parser.add_argument("--lora-r", type=int, default=8,
                       help="LoRA rank (lower = fewer params)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                       help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                       help="LoRA dropout")
    parser.add_argument("--target-modules", type=str, nargs="+", 
                       default=["query_key_value"],
                       help="Modules to apply LoRA to")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Output args
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*60}")
    print(f"LoRA Fine-tuning: {args.model}")
    print(f"{'='*60}")
    print(f"Dataset: {args.data}")
    print(f"Training samples: {args.samples}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Target modules: {args.target_modules}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    print(f"Loading dataset from {args.data}")
    dataset = load_from_disk(args.data)
    
    all_members = dataset["member"][:]
    train_samples = all_members[:args.samples]
    
    split_idx = int(len(train_samples) * 0.9)
    train_texts = train_samples[:split_idx]
    eval_texts = train_samples[split_idx:]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Eval samples: {len(eval_texts)}")
    
    test_start = args.samples
    test_end = test_start + 100
    test_members = all_members[test_start:test_end] if len(all_members) > test_end else all_members[-100:]
    test_nonmembers = dataset["nonmember"][:100]
    
    print(f"\nLoading base model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,  
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    print(f"Base model loaded with {model.num_parameters():,} total parameters.")
    
    print(f"\nConfiguring LoRA")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("\nTokenizing datasets")
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False 
        )
    
    train_tokenized = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing training data"
    )
    
    eval_tokenized = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing eval data"
    )
    
    print(f"Tokenization complete")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        warmup_steps=250,
        max_grad_norm=1.0,
        weight_decay=0.00,
        
        logging_steps=20,
        logging_first_step=True,
        eval_strategy="epoch",
        
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
        seed=args.seed,
        
        fp16=False,
        bf16=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        callbacks=[NaNDetectionCallback()],
    )
    
    print(f"\n{'='*60}")
    print(f"Starting LoRA training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    train_result = trainer.train()
    
    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(exist_ok=True)
    
    print(f"\nSaving LoRA adapter to {adapter_dir}")
    model.save_pretrained(adapter_dir)
    
    print(f"Merging LoRA weights with base model")
    trainable_params, total_params = model.get_nb_trainable_parameters()
    merged_model = model.merge_and_unload()
    
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    
    print(f"Saving merged model to {model_dir}")
    merged_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    lora_config_path = adapter_dir / "lora_config.json"
    with open(lora_config_path, 'w') as f:
        json.dump({
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "base_model": args.model,
        }, f, indent=2)
    
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    
    data_dir = output_dir / "data_split"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Saving test data to {data_dir}")
    with open(data_dir / "members.json", 'w') as f:
        json.dump(test_members, f, indent=2)
    
    with open(data_dir / "non_members.json", 'w') as f:
        json.dump(test_nonmembers, f, indent=2)
    
    metadata = {
        "base_model": args.model,
        "finetuning_method": "LoRA",
        "dataset": args.data,
        "num_train_samples": len(train_texts),
        "num_eval_samples": len(eval_texts),
        "lora_config": {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
        },
        "training_config": {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "final_train_loss": train_result.metrics.get("train_loss"),
        "final_eval_loss": train_result.metrics.get("eval_loss"),
        "trainable_params": model.get_nb_trainable_parameters(),
    }
    
    with open(data_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"LoRA Fine-tuning Complete")
    print(f"{'='*60}")
    print(f"LoRA adapter: {adapter_dir}")
    print(f"Merged model: {model_dir}")
    print(f"Test data: {data_dir}")
    print(f"Training metrics: {metrics_path}")
    print(f"\nFinal metrics:")
    print(f"  Train loss: {train_result.metrics.get('train_loss', 'N/A')}")
    print(f"  Eval loss: {train_result.metrics.get('eval_loss', 'N/A')}")
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    print(f"\nNext step:")
    print(f"  python run.py --config configs/lora_eval.json")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()