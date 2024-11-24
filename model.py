import argparse
import torch
import re
 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import setup_chat_format
from datasets import load_dataset
from peft import LoraConfig 
from trl import SFTTrainer
 
def main(args):
    # Base model id
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
 
    # Finetuned model id
    output_directory="./output/"
    peft_model_id=output_directory+"model"
     
    # Training data
    movie_output_final = [
        'output/final_scripts/american_psycho.jsonl',
        './output/final_scripts/no_time_to_die.jsonl',
        './output/final_scripts/top_gun_maverick.jsonl',
        './output/final_scripts/wedding_crashers.jsonl',
    ]
 
    # Load training data and split
    train_dataset = load_dataset("json", data_files=[file for file in movie_output_final], split="train")
    
    # Split dataset into 80-20%
    train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
    torch.utils.checkpoint.use_reentrant=True
 
    # Configure the Bits and Bites quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16  # Change from Niklas / Different from Phil Schmid's blog post
    )
 
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16, # Change from Niklas / Different from Phil Schmid's blog post
        quantization_config=bnb_config,
        return_dict=False
    )
    model.config.use_cache = False
 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'right'
    
    # Ensure the tokenizer's chat template is cleared before setup
    if hasattr(tokenizer, "chat_template"):
        tokenizer.chat_template = None  # Clear the existing template if present
 
    # Use for the training the chat format
    model, tokenizer = setup_chat_format(model, tokenizer)
 
    peft_config = LoraConfig(
            lora_alpha=128, 
            lora_dropout=0.05,
            r=32, # Change from Niklas / Different from Phil Schmid's blog post
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM"
    )
 
    args = TrainingArguments(
        output_dir=output_directory+"checkpoints", # The output directory where the model predictions and checkpoints will be written.
        logging_dir=output_directory+"logs", # Tensorboard log directory. Will default to runs/**CURRENT_DATETIME_HOSTNAME**.
        logging_strategy="steps",
        logging_steps=250,
        eval_strategy="steps", # Added by Thomas
        eval_steps=1000, # Added by Thomas
        save_steps=1000, # Number of updates steps before two checkpoint saves.
        num_train_epochs=3, # Total number of training epochs to perform.          
        per_device_train_batch_size=5, # The batch size per GPU/TPU core/CPU for training.
        gradient_accumulation_steps=2, # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},# Added by Thomas
        optim="adamw_torch_fused",   
        save_strategy="epoch",        
        learning_rate=2e-4, # The initial learning rate for Adam.
        fp16=True, # Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training. Change from Niklas / Different from Phil Schmid's blog post
        max_grad_norm=0.3, # Maximum gradient norm (for gradient clipping).                   
        warmup_ratio=0.03, # Number of steps used for a linear warmup from 0 to learning_rate.                  
        lr_scheduler_type="constant",          
        push_to_hub=False,  # Change from Niklas / Different from Phil Schmid's blog post               
        auto_find_batch_size=True # Change from Niklas / Different from Phil Schmid's blog post
    )
 
    # Supervised fine-tuning (or SFT for short) 
    max_seq_length = 512
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset['train'],
        eval_dataset=train_dataset['test'],
        peft_config=peft_config,
        max_seq_length=max_seq_length, # maximum packed length 
        tokenizer=tokenizer,
        dataset_kwargs={
            "add_special_tokens": False, 
            "append_concat_token": False,
        }
    )
 
    # Train the model
    trainer.train()
 
    # Save the model and tokenizer
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
 
    # Clean-up
    del model
    del trainer
    torch.cuda.empty_cache()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)