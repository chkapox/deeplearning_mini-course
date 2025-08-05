from torch import nn
from torch.nn import Sequential
import torch

class BaselineModel(nn.Module):
    def __init__(self, n_feats, n_class, fc_hidden=512):
        super().__init__()
        # MLP-путь (на случай входа [B, n_feats])
        self.net = nn.Sequential(
            nn.Linear(n_feats, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, n_class),
        )
        # LCNN-lite: MFM блоки + GAP
        self.cnn = nn.Sequential(
            MFM(1, 32, 3, 1, 1),
            MFM(32, 32, 3, 1, 1),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.1),

            MFM(32, 64, 3, 1, 1),
            MFM(64, 64, 3, 1, 1),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.1),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.cnn_head = nn.Linear(64, n_class)

    def forward(self, data_object, **batch):
        x = data_object
        if x.dim() == 4:                 # (B, 1, F, T)
            h = self.cnn(x).flatten(1)   # (B, 64)
            return {"logits": self.cnn_head(h)}
        elif x.dim() == 3:               # (B, F, T) -> усредним по времени
            x = x.mean(dim=-1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        return {"logits": self.net(x)}

class MFM(nn.Module):
    """Max-Feature-Map: разделяем каналы надвое и берём поэлементный максимум."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, stride, padding, bias=True)
        self.bn = nn.BatchNorm2d(out_ch * 2)

    def forward(self, x):
        x = self.bn(self.conv(x))
        c = x.shape[1] // 2
        a, b = x[:, :c, ...], x[:, c:, ...]
        return torch.maximum(a, b)


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


class MFM(nn.Module):
    """
    Max-Feature-Map активатор: делит каналы пополам и берет поэлементный max.
    Ожидает, что in_channels кратно 2.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, C, H, W), C = 2*k
        c = x.size(1)
        assert c % 2 == 0, "MFM expects even number of channels"
        a, b = x.split(c // 2, dim=1)
        return torch.max(a, b)


class ConvMFM(nn.Module):
    """
    Conv + MFM: свертка порождает в 2 раза больше каналов, MFM урезает их пополам.
    """
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=k, stride=s, padding=p, bias=bias)
        self.mfm  = MFM()

    def forward(self, x):
        return self.mfm(self.conv(x))


class LCNN(nn.Module):
    """
    Простая LCNN под вход (B, 1, F, T) с глобальным усреднением по пространству.
    Дает вектор признаков и классифицирует в 2 класса (spoof/bonafide).
    """
    def __init__(self, in_channels=1, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvMFM(in_channels, 64,  k=5, s=1, p=2),  # -> 64
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvMFM(64, 96, k=3, s=1, p=1),            # -> 96
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvMFM(96, 128, k=3, s=1, p=1),           # -> 128
            nn.MaxPool2d(kernel_size=2, stride=2),

            ConvMFM(128, 128, k=3, s=1, p=1),          # -> 128
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(128, num_classes)

        # инициализация
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data_object, **batch):
        """
        data_object: Tensor [B, 1, F, T]
        Возвращает словарь {'logits': [B, 2]}.
        """
        x = data_object
        if x.dim() == 3:
            # на всякий случай: [B, F, T] -> [B, 1, F, T]
            x = x.unsqueeze(1)

        x = self.features(x)                 # [B, C, H, W]
        x = x.mean(dim=[2, 3])               # Global Average Pool -> [B, C]
        x = self.dropout(x)
        logits = self.classifier(x)          # [B, 2]
        return {"logits": logits}

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
