# Transformer Translator

This project demonstrates how to train a Transformer model for translating sentences from English to Hungarian using the `opus_books` dataset.

## Project Structure

- `dataset.py`: Contains the `TranslationDataset` class and `load_datasets` function for preparing the data.
- `models.py`: Contains the `TransformerTranslator` model definition.
- `train.py`: Contains the training loop logic.
- `evaluate.py`: Contains the evaluation logic.
- `translate.py`: Contains a function to translate sentences using the trained model.
- `main.py`: The main script to tie everything together and run the training and evaluation.
- `README.md`: Project description and instructions.

## Instructions

1. Clone the repository.
2. Install the required packages:
    ```sh
    pip install pandas numpy torch transformers scikit-learn
    ```
3. Prepare your dataset in `test.csv`.
4. Run the training and evaluation:
    ```sh
    python main.py
    ```
5. Use `translate.py` to translate new sentences with the trained model.

## Example Usage

```python
from transformers import BertTokenizer
from models import TransformerTranslator
from translate import translate_sentence

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TransformerTranslator()
model.load_state_dict(torch.load('path_to_saved_model.pth'))

sentence = "Translate this sentence."
translated_sentence = translate_sentence(sentence, model, tokenizer)
print(translated_sentence)
```


This structure will make your project more organized and modular, making it easier to maintain and expand in the future.
