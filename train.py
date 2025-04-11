import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from model.data import CarDataset
from model.lstmlike import SalesPredictor
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt  

def main():
    device = torch.device('cuda')

    num_ep = 100
    bt = 4
    lr = 3e-6


    dataset = CarDataset()
    dataloader = DataLoader(
        dataset,
        bt,
        True
    )

    model = SalesPredictor()
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr, 
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    tl = []
    with tqdm(range(num_ep), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    x,y = nbatch
                    x = x.to(device)
                    y = y.to(device)

                    y_p=model(x)

                    loss = nn.functional.mse_loss(y_p, y.view(-1,1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                    # plt.figure()
                    # plt.plot(epoch_loss)
                    # plt.savefig(f"{bt}_epoch_{epoch_idx}.png")
                    # plt.close()
            lm = np.mean(epoch_loss)
            tglobal.set_postfix(loss=lm)
            tl.append(lm)
            plt.figure()
            plt.plot(tl)
            # plt.savefig(f"{bt}_ep_{lr}.png")
            plt.savefig(f"{bt}_ep_cos.png")
            plt.close()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                f"ckpt/{bt}_epoch_{epoch_idx}_{lm:.3f}.pt"
            )
            print("saving complete!")
if __name__ == "__main__":
    main()
