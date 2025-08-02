import torch
from torch import nn

class ExampleLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            w = torch.tensor(weight, dtype=torch.float)
        else:
            w = None
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits=None, labels=None, **batch):
        if logits is None: logits = batch["logits"]
        if labels is None: labels = batch["labels"]
        loss = self.ce(logits, labels.long())
        return {"loss": loss}
