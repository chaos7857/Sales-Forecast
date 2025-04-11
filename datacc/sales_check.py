import json 
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
with open("dataraw/sales_conclusion.json", "r", encoding="utf-8") as f:
    data = json.load(f)

a=['小米su7', '特斯拉cary', '问界m5', '特斯拉car3', '极氪007']

def format_time(t):
    year = int(t)
    month = int(round((t - year) * 100))
    return f"{year}-{month:02d}"

grouped = {}
for i in data:
    car = i["车型"]
    grouped.setdefault(car, []).append(i)
plt.figure(figsize=(12, 6))
for car, i in grouped.items():
    sorted_i = sorted(i, key=lambda x: x["时间"])
    times = [format_time(e["时间"]) for e in sorted_i]
    sales = [e["销量"] for e in sorted_i]
    plt.plot(times, sales, marker='o', linestyle='-', label=car)

plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

plt.show()
print("done .")

