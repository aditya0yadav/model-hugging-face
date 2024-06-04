# Use a pipeline as a high-level helper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import torch
import torch.nn.functional as F
# import tabula 
input = ""
with open('model.txt','r') as f :
    input+=f.read()
value_string = " ".join(input.split())
print(value_string)
    
# df = tabula.read_pdf("mofdel.pdf", pages='all')
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

warnings.filterwarnings("ignore")

input_text = value_string
encoded_input = tokenizer(input_text, return_tensors='pt')

output = model(**encoded_input)
print(output.logits)

value = output.logits.detach().numpy()
print(value)

logits = torch.tensor(value)
probabilities = F.softmax(logits, dim=0)
print(probabilities)
