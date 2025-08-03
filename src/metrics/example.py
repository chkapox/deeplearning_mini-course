import torch
from src.metrics.base_metric import BaseMetric

class ExampleMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)
        self._name = kwargs.get("name", self.metric.__class__.__name__)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        name = self._name.lower()
        labels = labels.long()

        if name == "eer":
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            y = labels.detach().cpu()
            sort_idx = torch.argsort(scores, descending=True)
            y_sorted = y[sort_idx]
            pos = (y_sorted == 1).to(torch.long)
            neg = (y_sorted == 0).to(torch.long)
            P = pos.sum().item()
            N = neg.sum().item()
            if P == 0 or N == 0:
                return 0.0
            tp = torch.cumsum(pos, 0).float()
            fp = torch.cumsum(neg, 0).float()
            tpr = tp / P
            fpr = fp / N
            fnr = 1 - tpr
            i = torch.argmin(torch.abs(fpr - fnr))
            eer = 0.5 * (fpr[i] + fnr[i])
            return float(eer)

        metric_name = self.metric.__class__.__name__
        if metric_name.startswith("Binary"):
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            return self.metric(prob_pos, labels)
        else:
            preds = logits.argmax(dim=1)
            return self.metric(preds, labels)
