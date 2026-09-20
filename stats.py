import json
from collections import Counter

predicted_counter = Counter()
predicted_correct = Counter()
actual_counter = Counter()
total = 0
correct = 0

with open("predictions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            rec = json.loads(line)
            total += 1
            p = rec["predicted"]
            a = rec["actual"]
            predicted_counter[p] += 1
            actual_counter[a] += 1
            if rec["correct"]:
                correct += 1
                predicted_correct[p] += 1
        except Exception:
            continue

print(f"Всего предсказаний: {total}")
print(f"Точность: {correct}/{total} ({correct/total*100:.1f}%)")
print()
print("РАСПРЕДЕЛЕНИЕ ПРЕДСКАЗАНИЙ (что ИИ ставил):")
for color, cnt in predicted_counter.most_common():
    hits = predicted_correct[color]
    acc = hits / cnt * 100 if cnt else 0
    base = actual_counter[color] / total * 100
    lift = (acc / base) if base > 0 else 0
    print(f"  {color}: {cnt} раз ({cnt/total*100:.1f}%) | "
          f"угадано {hits} ({acc:.1f}%) | "
          f"базовая частота {base:.1f}% | lift={lift:.2f}")
print()
print("РАСПРЕДЕЛЕНИЕ ФАКТА (что реально выпало):")
for color, cnt in actual_counter.most_common():
    print(f"  {color}: {cnt} ({cnt/total*100:.1f}%)")