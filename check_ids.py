import pandas as pd

# Пути
csv_path = "grading/students_solutions/dev_check.csv"
protocol_path = "data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

# Загружаем прогнозы
preds = pd.read_csv(csv_path, header=None, names=["utt_id", "score"])
print("Первые 5 строк CSV:")
print(preds.head())
print(f"Всего строк в CSV: {len(preds)}")

# Загружаем ID из протокола
with open(protocol_path, "r") as f:
    proto_ids = [line.strip().split()[0] for line in f]
print("\nПервые 5 ID из протокола:")
print(proto_ids[:5])
print(f"Всего строк в протоколе: {len(proto_ids)}")

# Считаем пересечение
overlap = set(preds["utt_id"]) & set(proto_ids)
print(f"\nСовпадающих ID: {len(overlap)}")
if len(overlap) > 0:
    print("Примеры совпадений:", list(overlap)[:10])
else:
    print("Совпадений нет — значит CSV и протокол относятся к разным сплитам или формат ID не совпадает.")
