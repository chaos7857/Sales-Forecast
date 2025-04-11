import json
with open("datacc\dong\ZEEKR.json", "r", encoding="utf-8") as f:
    data = json.load(f)
a = set()
for i in data:
    a.add(i["car"][:5])#7

for x in a:
    res = []
    for i in data:
        if(i["car"][:5]==x and i["time"] is not None):
            res.append(i)
    with open(f"{x}.json","w",encoding="utf-8") as t:
        json.dump(res,t,ensure_ascii=False)
print("done .")