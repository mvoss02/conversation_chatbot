from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
peft_model_id = "./output/model"  # Path to your fine-tuned model
model = AutoModelForCausalLM.from_pretrained(peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

# Read the prompt from a text file
with open('prompt.txt', 'r') as file:
    prompt = file.read().strip()  # Reads the content and removes any extra spaces/newlines

# Display the loaded prompt
print(f"Loaded Prompt: {prompt}")

# Tokenize the input
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

# Generate response
output = model.generate(
    input_ids,
    max_length=200,             # Maximum length of the output
    temperature=0.8,            # Adjust this for more or less creativity
    top_p=0.9,                  # Nucleus sampling for diversity
    repetition_penalty=1.2,     # To reduce repetitive outputs
    do_sample=True              # Enable sampling
)

# Decode and print the result
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Response:", response)