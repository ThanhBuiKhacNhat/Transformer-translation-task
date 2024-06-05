import torch.nn as nn
from transformers import BertModel, BertConfig

class TransformerTranslator(nn.Module):
    def __init__(self):
        super(TransformerTranslator, self).__init__()
        self.config = BertConfig.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', config=self.config)
        self.fc = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state)
        return logits
