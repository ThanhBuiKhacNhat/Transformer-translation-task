import torch
from model.transformer import TransformerTranslator

def translate(model, sentence, input_vocab, target_vocab, device, max_length=50):
    model.eval()

    # Preprocess the sentence
    sentence = [input_vocab.stoi["<sos>"]] + [input_vocab.stoi[word] for word in sentence.split()] + [input_vocab.stoi["<eos>"]]
    sentence = torch.LongTensor(sentence).unsqueeze(0).to(device)

    # Create masks
    input_mask = (sentence != input_vocab.stoi["<pad>"]).to(device)

    with torch.no_grad():
        output = model(sentence, sentence, input_mask, input_mask)

    output = output.argmax(dim=-1)

    translated_sentence = [target_vocab.itos[idx] for idx in output[0]]
    # Remove <sos> and <eos> tokens
    translated_sentence = translated_sentence[1:-1]

    return ' '.join(translated_sentence)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerTranslator(num_layers, d_model, num_heads, hidden_dim, input_vocab_size, target_vocab_size, max_seq_len, dropout, learning_rate, batch_size)
model.load_state_dict(torch.load('path/to/model.pth'))
model = model.to(device)

# Assume input_vocab and target_vocab are the vocabulary for the input and target languages respectively
sentence = "This is a test sentence."
translated_sentence = translate(model, sentence, input_vocab, target_vocab, device)
print(f"Translated Sentence: {translated_sentence}")