from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load BERT model and tokenizer for sentiment analysis
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define a function for sentiment analysis
def analyze_sentiment(text):
    # Tokenize input text
    encoded_input = tokenizer(text, return_tensors='pt')
    
    # Perform model inference
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(output.logits, dim=1)
    
    # Get predicted sentiment label
    sentiment_label = "Positive" if probabilities[0][1] > probabilities[0][0] else "Negative"
    
    # Return sentiment label and probabilities
    return sentiment_label, probabilities[0][1].item(), probabilities[0][0].item()

# Example text for sentiment analysis
text = "I really enjoyed the movie! It was fantastic."

# Perform sentiment analysis
sentiment_label, positive_probability, negative_probability = analyze_sentiment(text)

# Print results
print("Text:", text)
print("Predicted Sentiment:", sentiment_label)
print("Positive Probability:", positive_probability)
print("Negative Probability:", negative_probability)
