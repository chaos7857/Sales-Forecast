import torch
from torch.utils.data import DataLoader
from model.data import CarDataset
from model.lstmlike import SalesPredictor
import numpy as np
from tqdm.auto import tqdm

def evaluate(checkpoint_path, batch_size=2):
    device = torch.device('cuda')
    model = SalesPredictor()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()
    
    test_dataset = CarDataset(False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False 
    )
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        with tqdm(test_dataloader, desc='Evaluating') as tepoch:
            for x, y in tepoch:
                x = x.to(device)
                y = y.to(device)
                
                y_pred = model(x)
                
                all_predictions.append(y_pred.cpu().numpy())
                all_targets.append(y.view(-1, 1).cpu().numpy())
                
                loss = torch.nn.functional.mse_loss(y_pred, y.view(-1, 1))
                total_loss += loss.item() 
    
    avg_loss = total_loss / len(test_dataset)
    
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    print(f'\nEvaluation Results:')
    print(f'MSE Loss: {avg_loss:.4f}')
    
    return {
        'loss': avg_loss,
        'predictions': all_predictions,
        'targets': all_targets
    }
import os

def eval(checkpoint_path= "ckpt/16_epoch_5_0.293.pt"):
    results = evaluate(checkpoint_path)
    
    
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
    
    sorted_i = sorted(grouped['特斯拉modely'], key=lambda x: x["时间"])
    times = [format_time(e["时间"]) for e in sorted_i]
    sales = [e["销量"] for e in sorted_i]
    plt.plot(times[:-1], sales[:-1], marker='o', linestyle='-', label='特斯拉modely')
    
    plt.axvline(x=22,color='r', linestyle='-.', linewidth=2)
    
    p1=np.mean(results["predictions"][:50])+1
    p2=np.mean(results["predictions"][50:])+1
    yy1=p1*sales[22]
    yy2=p2*yy1
    plt.plot([22,23,24],[sales[22],yy1,yy2],color='y', linestyle='-.', linewidth=2)

    plt.title(checkpoint_path+'-'+str(results["loss"]))
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    global t
    plt.savefig(f"sale{t}.png")
    t+=1
    print("done .")


if __name__ == "__main__":
    a =os.listdir('ckpt')
    global t 
    t=0
    for i in a:
        pa = "ckpt/"+i
        eval(pa)