"""
Fine-tuning Script for Qwen3:8b using LoRA

This script implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation)
for the Qwen3:8b model. LoRA is the industry standard approach for fine-tuning LLMs because it:

1. Reduces trainable parameters by ~99% (only trains small adapter layers)
2. Requires significantly less GPU memory
3. Trains faster than full fine-tuning
4. Can be easily merged back into the base model or swapped

Technical Details:
- Uses Hugging Face transformers and PEFT libraries
- Implements LoRA with configurable rank and alpha parameters
- Supports gradient checkpointing for memory efficiency
- Uses bitsandbytes for 4-bit quantization (QLoRA) to reduce memory further (Linux/Windows only)
- Includes proper evaluation and checkpoint saving
"""

# IMPORTANT
# Visit https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_qwen3 for more information
# Also, visit https://www.datacamp.com/tutorial/fine-tuning-qwen3

import torch
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from datasets import Dataset

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if bitsandbytes is available (not available on macOS)
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    BITSANDBYTES_AVAILABLE = True
except (ImportError, AttributeError):
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes not available - 4-bit quantization will be disabled")
    logger.warning("This is expected on macOS. Training will use FP16 instead.")


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning hyperparameters."""

    model_name: str = "Qwen/Qwen3-8B"
    output_dir: str = "fine_tuning/output"

    # LoRA hyperparameters
    lora_rank: int = 8  # Rank of the low-rank matrices (higher = more capacity, slower)
    lora_alpha: int = 16  # Scaling parameter (typically 2x rank)
    lora_dropout: float = 0.05  # Dropout for regularization
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",  # Attention matrices
            "gate_proj",
            "up_proj",
            "down_proj",  # MLP layers
        ]
    )

    # Training hyperparameters
    num_epochs: int = 3  # Number of training passes through the dataset
    batch_size: int = 1  # Batch size per GPU (keep small for 8B model)
    gradient_accumulation_steps: int = 4  # Effective batch size = batch_size * this
    learning_rate: float = 2e-4  # Learning rate for AdamW optimizer
    max_seq_length: int = 512  # Maximum sequence length
    warmup_steps: int = 10  # Learning rate warmup

    # Quantization settings (QLoRA) - only works on Linux/Windows
    use_4bit: bool = BITSANDBYTES_AVAILABLE  # Auto-disable on macOS
    bnb_4bit_compute_dtype: str = "float16"  # Computation dtype
    bnb_4bit_quant_type: str = "nf4"  # Quantization type (nf4 is optimal)

    # Data settings
    dataset_path: str = "data/training_dataset.jsonl"

    # System settings
    seed: int = 42


def load_dataset_from_jsonl(file_path: str) -> Dataset:
    """
    Load and prepare the training dataset from JSONL file.

    Args:
        file_path: Path to the JSONL file containing training examples

    Returns:
        Hugging Face Dataset object
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return Dataset.from_list(data)


def format_instruction(example: Dict) -> Dict:
    """
    Format examples into the instruction-following format expected by Qwen models.

    Qwen models expect a specific chat format. We convert our dataset into this format
    with clear system, user, and assistant roles.

    Args:
        example: Dictionary with 'instruction', 'input', 'output', and 'system' keys

    Returns:
        Dictionary with formatted 'text' field ready for training
    """
    # Build the conversation in Qwen's expected format
    system_message = example.get("system", "")
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    # Create the formatted prompt
    if input_text:
        prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{instruction}

Input: {input_text}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    else:
        prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""

    return {"text": prompt}


def setup_model_and_tokenizer(config: FineTuningConfig):
    """
    Initialize the model with quantization and LoRA configuration.

    This function:
    1. Loads the tokenizer
    2. Configures 4-bit quantization (QLoRA) if available
    3. Loads the base model
    4. Prepares it for k-bit training (if quantization enabled)
    5. Applies LoRA adapters

    Args:
        config: Fine-tuning configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Setting up model and tokenizer...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer loaded: {config.model_name}")

    # Configure 4-bit quantization (QLoRA) - only if bitsandbytes is available
    bnb_config: Optional[object] = None
    if config.use_4bit and BITSANDBYTES_AVAILABLE:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("4-bit quantization configured (QLoRA)")
    elif not BITSANDBYTES_AVAILABLE:
        logger.info("Training will use FP16 (quantization not available)")

    # Load base model
    device_map_config = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map=device_map_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False if sys.platform == "darwin" else True,
    )

    logger.info(f"Base model loaded: {config.model_name}")

    # Move model to MPS if available and not using CUDA
    if torch.backends.mps.is_available() and not torch.cuda.is_available():
        model = model.to("mps")
        logger.info("Model moved to MPS device")

    # Prepare model for k-bit training (only if quantization enabled)
    if bnb_config is not None and BITSANDBYTES_AVAILABLE:
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for k-bit training")
    else:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA adapters applied - Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%), Total: {total_params:,}")

    return model, tokenizer


def train(config: FineTuningConfig):
    """
    Main training function that orchestrates the entire fine-tuning process.

    Args:
        config: Fine-tuning configuration
    """
    logger.info("QWEN3:8B FINE-TUNING WITH LORA")

    # Print platform info
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.backends.mps.is_available():
        logger.info(f"MPS (Metal) available: True")
    logger.info(f"Quantization: {'Enabled (4-bit)' if config.use_4bit and BITSANDBYTES_AVAILABLE else 'Disabled (FP16)'}")

    # Set random seed for reproducibility
    torch.manual_seed(config.seed)

    # Load and prepare dataset
    logger.info("Loading dataset...")
    raw_dataset = load_dataset_from_jsonl(config.dataset_path)
    logger.info(f"Loaded {len(raw_dataset)} training examples")

    # Format dataset for training
    logger.info("Formatting dataset...")
    formatted_dataset = raw_dataset.map(
        format_instruction,
        remove_columns=raw_dataset.column_names,
    )

    # Initialize model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # Tokenize the dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    logger.info(f"Dataset tokenized ({len(tokenized_dataset)} examples)")

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=1,  # Log every step (useful for small datasets)
        save_strategy="epoch",  # Save checkpoint after each epoch
        save_total_limit=2,  # Keep only the 2 most recent checkpoints
        fp16=not sys.platform == "darwin",  # Use FP16 on Linux/Windows
        optim="adamw_torch",  # Optimizer
        report_to="none",  # Disable wandb/tensorboard
        seed=config.seed,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Start training
    logger.info("Starting training...")
    logger.info(f"Total training examples: {len(tokenized_dataset)}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    logger.info(
        f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}"
    )
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info("=" * 60)

    trainer.train()

    # Save the fine-tuned model
    logger.info("Training complete! Saving model...")
    final_model_path = output_dir / "final_model"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    logger.info(f"Model saved to: {final_model_path}")
    logger.info("To use the fine-tuned model:")
    logger.info("  from peft import PeftModel")
    logger.info("  from transformers import AutoModelForCausalLM")
    logger.info("  ")
    logger.info(f"  base_model = AutoModelForCausalLM.from_pretrained('{config.model_name}')")
    logger.info(f"  model = PeftModel.from_pretrained(base_model, '{final_model_path}')")

    # Save configuration for reference
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_name": config.model_name,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "quantization": "4-bit" if config.use_4bit and BITSANDBYTES_AVAILABLE else "FP16",
                "platform": sys.platform,
            },
            f,
            indent=2,
        )

    logger.info(f"Training configuration saved to: {config_path}")


def main():
    """
    Main entry point for the fine-tuning script.

    Usage:
        python fine_tuning/training.py
    """
    config = FineTuningConfig()

    if not Path(config.dataset_path).exists():
        logger.error(f"Dataset not found at {config.dataset_path}")
        logger.error("Please run dataset_preparation.py first to generate the training data.")
        return

    train(config)


if __name__ == "__main__":
    main()
