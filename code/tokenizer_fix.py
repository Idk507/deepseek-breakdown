# Add this code before the line that's causing the error:

from transformers import BertTokenizer

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Now you can use the tokenizer
sentence = "The artist painted the portrait of a woman with a brush"
inputs = tokenizer(sentence, return_tensors='pt')
tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']  # Add special tokens

print("Tokens:", tokens)
print("Input IDs:", inputs['input_ids'])