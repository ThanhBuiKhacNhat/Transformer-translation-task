import torch

def translate_sentence(model, tokenizer, sentence, max_length=128):
    """
    Translates a given sentence using the trained model.
    
    Args:
        model (nn.Module): The trained model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding and decoding sentences.
        sentence (str): The input sentence to be translated.
        max_length (int): The maximum sequence length for the tokenizer.

    Returns:
        str: The translated sentence.
    """
    model.eval()
    tokens = tokenizer(sentence, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = tokens['input_ids'].to(model.device)
    attention_mask = tokens['attention_mask'].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    _, predicted = torch.max(outputs, 2)
    predicted_ids = predicted.squeeze(0).tolist()
    translated_sentence = tokenizer.decode(predicted_ids, skip_special_tokens=True)
    
    return translated_sentence
