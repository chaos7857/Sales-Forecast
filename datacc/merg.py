import os
import  json

def merg_by_car():
    # a  = set()
    res = {}
    x = os.listdir("datacc")
    for file in x:
        fold = os.path.join("datacc",file)
        if os.path.isdir(fold):
            xx = os.listdir(fold)
            for xxfile in xx:
                # print(os.path.join(fold,xxfile))#{'比亚迪 汉.json', '小米su7.json', '汉DM.json', 'Model 3.json', '极氪007.json', 'ZEEKR.json', 'Model Y.json', '问界M5.json'}
                # a.add(xxfile)
                with open(os.path.join(fold,xxfile), "r", encoding="utf-8") as f:
                    data = json.load(f)
                res.setdefault(xxfile, []).extend(data)


    # for k,v in res.items():
    #     for i in v:
    #         if(i["time"]==''):
    #             continue
    #         i["time"]=i["time"].split('-')[0]+'.'+i["time"].split('-')[1]
    #     v=sorted(v,key=lambda x:x['time'])


    for k, v in res.items():
        # 使用列表推导式同时完成：过滤空值 + 格式转换
        filtered_and_processed = [
            {"time": item["time"].split('-')[0]+'.'+item["time"].split('-')[1], **{ky: vl for ky, vl in item.items() if ky != "time"}}
            for item in v 
            if item.get("time", "") != ''  # 过滤空时间记录
        ]
        
        # 按时间排序（转换为浮点数保证数值排序准确）
        res[k] = sorted(
            filtered_and_processed,
            key=lambda x: float(x["time"])
        )

    
    print("done .")


if __name__=="__main__":
    merg_by_car()