import argparse
import multiprocessing
import os
from typing import Optional
from pathlib import Path

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
    AutoTokenizer,
)
from trl import SFTTrainer
import traceback


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--dataset_name", type=str, default="the-stack-smol")
    parser.add_argument("--subset", type=str, default="data/rust")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset_text_field", type=str, default="content")
    
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_training_steps", type=int, default=1000)
    parser.add_argument("--num_cycles", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_starcoder2")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory or 'latest' to use the latest checkpoint",
    )

    parser.add_argument("--push_frequency", type=int, default=500)
    parser.add_argument("--hub_repo_id", type=str, required=True)
    
    return parser.parse_args()


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_latest_checkpoint(output_dir):
    """
    Returns the path to the latest checkpoint in the output directory.
    """
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
    return latest_checkpoint


def calculate_dataset_tokens(dataset, tokenizer, text_field="content"):
    """
    Calculate the total number of tokens in the dataset.
    """
    total_tokens = 0
    for item in dataset:
        tokens = tokenizer(item[text_field], return_length=True)["length"][0]
        total_tokens += tokens
    
    return total_tokens


def sample_dataset_by_batch(dataset, max_steps, batch_size, gradient_accumulation_steps, seed=42):
    """
    Sample dataset based on batch size and gradient accumulation steps.
    """
    # Calculate total samples needed for training
    samples_per_step = batch_size * gradient_accumulation_steps
    total_samples_needed = samples_per_step * max_steps
    
    if total_samples_needed >= len(dataset):
        print(f"Using full dataset: {len(dataset)} examples")
        return dataset
    
    # Sample the dataset
    sampled_dataset = dataset.shuffle(seed=seed).select(range(total_samples_needed))
    print(f"Sampled {total_samples_needed} examples from {len(dataset)} total examples")
    print(f"Using {samples_per_step} samples per step for {max_steps} steps")
    return sampled_dataset


def safe_push_to_hub(trainer, message):
    """
    Safely push model to hub with proper error handling
    """
    try:
        # First try to save the model locally
        print("Saving model locally...")
        trainer.save_model()
        
        # Then attempt to push to hub
        print("Pushing to hub...")
        trainer.push_to_hub(message, create_model_card=False)  # Disable automatic model card creation
        
        # Create a basic model card manually
        readme_content = f"""---
language: code
tags:
- trl
- causal-lm
- text-generation
pipeline_tag: text-generation
---

# Model Details

This is a fine-tuned version of {trainer.args.hub_model_id} using TRL's SFTTrainer.

## Training Setup
- Learning rate: {trainer.args.learning_rate}
- Batch size: {trainer.args.per_device_train_batch_size}
- Max steps: {trainer.args.max_steps}
- Weight decay: {trainer.args.weight_decay}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{trainer.args.hub_model_id}")
tokenizer = AutoTokenizer.from_pretrained("{trainer.args.hub_model_id}")
```
"""
        
        # Push the README separately
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=readme_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=trainer.args.hub_model_id,
                token=os.environ.get("HF_TOKEN"),
            )
            print("Successfully pushed model card to hub")
        except Exception as e:
            print(f"Warning: Failed to push model card, but model was uploaded successfully: {str(e)}")
            
    except Exception as e:
        print(f"Error pushing to hub: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        print("\nAttempting to save model locally only...")
        try:
            trainer.save_model()
            print("Successfully saved model locally")
        except Exception as local_e:
            print(f"Error saving model locally: {str(local_e)}")
            traceback.print_exc()


def main(args):
    # Load tokenizer for token counting
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit            =True,
        bnb_4bit_quant_type     ="nf4",
        bnb_4bit_compute_dtype  =torch.bfloat16,
    )
    
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    if args.resume_from_checkpoint == "latest":
        args.resume_from_checkpoint = get_latest_checkpoint(args.output_dir)
        print(f"Resuming from latest checkpoint: {args.resume_from_checkpoint}")
    elif args.resume_from_checkpoint:
        print(f"Resuming from specified checkpoint: {args.resume_from_checkpoint}")

    token = os.environ.get("HF_TOKEN", None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": PartialState().process_index},
        attention_dropout=args.attention_dropout,
        token=token,
        trust_remote_code=True
    )
    print_trainable_parameters(model)

    print("Loading dataset...")
    data = load_dataset(
        args.dataset_name,
        data_dir    =   args.subset,
        split       =   args.split,
        token       =   token,
        num_proc=args.num_proc if args.num_proc else multiprocessing.cpu_count(),
    )
    
    # Sample the dataset based on batch settings
    data = sample_dataset_by_batch(
        data,
        max_steps=args.max_steps,
        batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed
    )

    # Calculate and print token statistics
    total_tokens = calculate_dataset_tokens(data, tokenizer, args.dataset_text_field)
    print(f"\nDataset Statistics:")
    print(f"Total number of tokens: {total_tokens:,}")
    print(f"Average tokens per sample: {total_tokens / len(data):,.2f}")
    print(f"Total samples: {len(data)}")

    cycle_length = args.num_training_steps // args.num_cycles

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        max_seq_length=args.max_seq_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size =   args.micro_batch_size,
            gradient_accumulation_steps =   args.gradient_accumulation_steps,
            warmup_steps                =   args.warmup_steps,
            max_steps                   =   args.max_steps,
            learning_rate               =   args.learning_rate,
            lr_scheduler_type           =   "cosine_with_restarts",
            weight_decay                =   args.weight_decay,
            bf16                        =   args.bf16,
            logging_strategy            =   "steps",
            logging_steps               =   10,
            output_dir                  =   args.output_dir,
            optim                       =   "paged_adamw_8bit",
            seed                        =   args.seed,
            run_name                    =   f"train-{args.model_id.split('/')[-1]}",
            report_to                   =   "wandb",
            save_strategy               =   "steps",
            save_steps                  =   args.save_steps,
            save_total_limit            =   args.save_total_limit,
            hub_strategy                =   "every_save",
            hub_model_id                =   args.hub_repo_id,
            hub_token                   =   os.environ.get("HF_TOKEN"),
            push_to_hub                 =   args.push_to_hub,
        ),
        peft_config                     =   lora_config,
        dataset_text_field              =   args.dataset_text_field,
    )

    print("Training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("Saving the last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    
    if args.push_to_hub:
        safe_push_to_hub(trainer, "Final model upload")
    
    print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)


