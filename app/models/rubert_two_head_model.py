from torch import nn
from transformers import AutoModel

class RuBertTwoHeadFearModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.2)

        self.has_fear_head = nn.Linear(hidden_size, 1)

        self.fear_level_head = nn.Linear(hidden_size, 4)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        x = self.dropout(cls_embedding)

        has_fear_logits = self.has_fear_head(x).squeeze(-1)

        fear_level_logits = self.fear_level_head(x)

        return has_fear_logits, fear_level_logits