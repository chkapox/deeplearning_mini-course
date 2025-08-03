import torch
from torch import nn

class ExampleLoss(nn.Module):
    def __init__(self, weight=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__()
        w = None
        if weight is not None:
            w = torch.tensor(weight, dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)

    def forward(self, logits, labels, **batch):
        loss = self.ce(logits, labels.long())
        return {"loss": loss}
