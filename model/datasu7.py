from torch.utils.data import Dataset
import torch
import json
from sklearn.preprocessing import StandardScaler
from transformers import BertModel, BertTokenizer

from model.wjyhelper import DataHelper

class CarDataset(Dataset):
    def __init__(self,istrain=True):
        super().__init__()
        self.helper = DataHelper()
        self.comm = self.helper.get_XiaoMIsu7_all_comments()[27:36]
        self.scaler = StandardScaler()
        BERT_PATH = "./chinese-bert-wwm"
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        with open("dataraw/sales_conclusion.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        grouped = {}
        for i in data:
            car = i["车型"]
            grouped.setdefault(car, []).append(i)
        sale = grouped['小米su7'][15:25]
        self.sale = []
        for i in sale:
            self.sale.append(i['销量'])

        self.sale = torch.tensor(self.sale, dtype=torch.float32).reshape(-1,1)
        

        self.data = []
        num = []
        for i in range(9):
            comm =list(self.comm[i].values())[0]
            tem =int((len(comm)-10))
            num.append(tem)
            for j in range(tem):
                self.data.append((
                    comm[j:j+10],
                    torch.tensor((self.sale[i+1]-self.sale[i])/self.sale[i], dtype=torch.float32),
                ))
        #--------------------------------------------------------------------------------------------------   
        # self.sale2=[]
        # for i in grouped["小米su7"]:
        #     self.sale2.append(i['销量'])
        # self.sale2 = self.sale2[15:]
        # self.sale2 = torch.tensor(self.sale2, dtype=torch.float32).reshape(-1,1)
        # self.comm2 = self.helper.get_XiaoMIsu7_all_comments()[26:39]
        print("init over")

    def __len__(self):
        print(len(self.data))
        return len(self.data)
    
    def __getitem__(self, index):
        # len
        x = self.tokenizer(
            self.data[index][0],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        # len*512
        # y=self.scaler.fit_transform(self.data[index][1].reshape(-1,1))
        y=self.data[index][1].reshape(-1,1).repeat(10,1)
        return x,y



if __name__ == "__main__":
    c = CarDataset()
    x, y = c.__getitem__(1)
    print("done .")