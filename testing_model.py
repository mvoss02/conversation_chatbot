from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import torch
from trl import setup_chat_format
from helper import config

# Define the path to the fine-tuned model
peft_model_id = "./output/model"  # Your fine-tuned model path

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16  # Change from Niklas / Different from Phil Schmid's blog post
    )

# Load the model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, device_map="auto", torch_dtype=torch.float16, quantization_config=bnb_config,)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
tokenizer.padding_side = 'right'

model.resize_token_embeddings(len(tokenizer))
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
# Ensure the tokenizer's chat template is cleared before setup
if hasattr(tokenizer, "chat_template"):
    tokenizer.chat_template = None  # Clear the existing template if present

# Use for the training the chat format
model, tokenizer = setup_chat_format(model, tokenizer)

# Set model to evaluation mode (important for inference)
model.eval()

special_tokens = {
    "pad_token": "[PAD]",
    "eos_token": "</s>",
    "bos_token": "<s>",
    "additional_special_tokens": ["<user>", "<assistant>"]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# Scenario 1: Generate an opener (no previous messages)
person_info = {'name': 'Anna', 'age': 28, 'job': 'Photographer'}
my_info = {'name': 'John', 'age': 32, 'job': 'Software Engineer'}

prompt = "<user>: Generate a short, flirty, and funny opener to start a conversation with Anna.\n<assistant>:"

# Tokenize the input
messages = [
    {
        "role": "system",
        "content": config.SYSTEM_MESSAGE + "Generate a short and fun opener for a women called Anna who is a Photographer.",
    },
    {"role": "user", "content": ""},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

# Generate the response
output = model.generate(
    input_ids=tokenized_chat,
    max_length=512,  # Limit output length to a single response
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id, # End generation at the eos_token
)

# Decode and print the result
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Response:", response)