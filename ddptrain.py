import os
import argparse
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.data import CarDataset
from model.lstmlike import SalesPredictor


def ddpsetup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10086'
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def main(rank, args):
    print("train start")
    ddpsetup(rank, args.world_size)

    ckpt__path = args.ckpt_path
    if not os.path.exists(ckpt__path):
        os.makedirs(ckpt__path)
    print(f"ckpt will be save in {ckpt__path}")

    num_ep = args.ep
#------------------------------------------------------------
    model = SalesPredictor()

    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # 暂时不用了吧
    # ema = EMAModel(parameters=model.parameters(),power=0.75)
    # lr_scheduler

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr, 
        # weight_decay=1e-6
    )

    dataset = CarDataset(

    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) 
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.bt,
        num_workers=4,
        sampler=sampler
    )

    device = torch.device('cuda')
    model.to(device)
    if rank == 0:
        print("ok")
    #------------------------------------------------------------
    with tqdm(range(num_ep), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            num=0
            dataloader.sampler.set_epoch(epoch_idx)
            #------------------------------------------------------------
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    x0,x1,x2,x3,y = nbatch
                    x0.to(device)
                    y_p = model(x0,x1,x2,x3)
                    loss = nn.functional.mse_loss(y_p, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

                    if rank == 0:
                        plt.figure()
                        plt.plot(epoch_loss)
                        plt.savefig(f"loss_epoch{epoch_idx}.png")
                        plt.close()
                        num +=1
            #------------------------------------------------------------each batch
            loss_mean = np.mean(epoch_loss)
            tglobal.set_postfix(loss=loss_mean)
            if rank == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    f"./ckpt/model-{epoch_idx}-{loss_mean:.3f}.pt")
                print("saving complete!")
    #------------------------------------------------------------each epoch
#------------------------------------------------------------
    dist.destroy_process_group()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='configurating the extraction specifications')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--ckpt_path', default="ckpt")
    parser.add_argument('--ep', default="10")
    parser.add_argument('--lr', default="2e-5")
    parser.add_argument('--bt', default="16")
    args = parser.parse_args()
    # main(args)
    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.world_size, join=True)