from torch import nn
from torch.nn import Sequential
import torch


class BaselineModel(nn.Module):
    def __init__(self, n_feats, n_class, fc_hidden=512):
        super().__init__()
        # MLP-путь (для векторов [B, n_feats])
        self.net = Sequential(
            nn.Linear(n_feats, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, n_class),
        )
        # CNN-путь (для карт [B,1,F,T])
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),   # глобальный пул по F и T
        )
        self.cnn_head = nn.Linear(32, n_class)
        
    def forward(self, data_object, **batch):
        x = data_object
        if x.dim() == 4:              # (B,1,F,T)
            h = self.cnn(x).squeeze(-1).squeeze(-1)  # (B,32)
            logits = self.cnn_head(h)
            return {"logits": logits}
        elif x.dim() == 3:            # (B,F,T) -> усредним по времени
            x = x.mean(dim=-1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        # дальше MLP-путь
        return {"logits": self.net(x)}


    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class MFM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        c = self.channels
        x1, x2 = x[:, :c], x[:, c:]
        return torch.maximum(x1, x2)


class ConvMFMBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.mfm = MFM(out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mfm(x)
        x = self.pool(x)
        return x


class LCNN(nn.Module):
    def __init__(self, n_class: int = 2, base_ch: int = 64, dropout: float = 0.2):
        super().__init__()
        self.front = nn.Sequential(
            ConvMFMBlock(1, base_ch, k=5, p=2, pool=True),
            ConvMFMBlock(base_ch, 128, k=3, p=1, pool=True),
            ConvMFMBlock(128, 256, k=3, p=1, pool=True),
            ConvMFMBlock(256, 256, k=3, p=1, pool=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_class),
        )

    def forward(self, data_object, **batch):
        x = self.front(data_object)
        x = self.pool(x)
        logits = self.classifier(x)
        return {"logits": logits}
