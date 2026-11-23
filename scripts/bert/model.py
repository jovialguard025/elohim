import torch
import torch.nn as nn
from transformers import AutoModel

class SpanClassifier(nn.Module):
    def __init__(self, name="microsoft/deberta-v3-large", n_extra=0):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h * 5 + n_extra, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3)
        )

    @staticmethod
    def pool(H, m):
        m = m.unsqueeze(-1).to(H.dtype)
        return (H * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask, a_mask, b_mask, p_mask, extra=None):
        H = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        A = self.pool(H, a_mask)
        B = self.pool(H, b_mask)
        P = self.pool(H, p_mask)
        x = torch.cat([A, B, P, A * P, B * P], dim=-1)
        if extra is not None:
            x = torch.cat([x, extra], dim=-1)
        return self.head(x)
