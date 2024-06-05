import torch
from transformers import BertTokenizer
from models import TransformerTranslator

def translate_sentence(sentence, model, tokenizer, max_length=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask)
        predicted_ids = torch.argmax(outputs, dim=2)
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.squeeze().tolist())
        translated_sentence = tokenizer.convert_tokens_to_string(predicted_tokens)
        
    return translated_sentence
