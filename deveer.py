import pandas as pd
from sklearn.metrics import roc_curve
import numpy as np

csv_path = "grading/students_solutions/dev_neg.csv"
protocol_path = "data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

# Загружаем прогнозы
preds = pd.read_csv(csv_path, header=None, names=["utt_id", "score"])

# Загружаем протокол
labels = []
with open(protocol_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        utt_id = parts[1]  # это LA_T_**** — правильный ID
        label_str = parts[-1].lower()
        label = 1 if "bona" in label_str else 0
        labels.append((utt_id, label))

labels_df = pd.DataFrame(labels, columns=["utt_id", "label"])

# Склеиваем по utt_id
merged = preds.merge(labels_df, on="utt_id", how="inner")

print(f"Совпавших строк: {len(merged)} из {len(preds)}")

# Убедимся, что score — это float
merged["score"] = merged["score"].astype("float32")

# Вычисляем EER
fpr, tpr, thresholds = roc_curve(merged["label"], merged["score"], pos_label=1)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer_threshold = thresholds[eer_idx]
eer = fpr[eer_idx]

print(f"EER на dev: {eer*100:.4f}% (порог={eer_threshold:.4f})")
