import torch
from torch import nn
from transformers import BertModel
from transformers import BertModel, BertTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SalesPredictor(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        BERT_PATH = "./chinese-bert-wwm"
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.hidden_size = hidden_size
        self.comm_fc = nn.Linear(768, hidden_size)
        
        self.his_fc = nn.Linear(1, hidden_size)
        
        # self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

        encoder_layer = TransformerEncoderLayer(
            d_model=32, 
            nhead=4,
            dim_feedforward=128,
            dropout=0.1
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2)


        decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=12)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.output_fc = nn.Linear(hidden_size, 1) 
        self.sigmod = nn.Sigmoid()
        # self.ln = nn.LayerNorm()
    def forward(self, x):
        with torch.no_grad():
            bert_output = self.bert(
                input_ids=x["input_ids"].view(-1,512),
                attention_mask=x["attention_mask"].view(-1,512)
            )
        comm_fea = bert_output.last_hidden_state[:, 0, :]
        comm_fea = self.comm_fc(comm_fea)
        # comm_fea=self.ln(comm_fea)
        # other_f = self.numeric_fc(x)
        
        # combined = torch.stack([comm_fea, numeric_features], dim=1)
        # attn_output, _ = self.attention(combined, combined, combined)
        # fused = torch.mean(attn_output, dim=1)  # [batch, hidden]
        # temp = comm_fea
        comm_fea=self.decoder(tgt=comm_fea,memory=comm_fea)
        # comm_fea = temp + comm_fea

        # self.encoder

        # lstm_out, _ = self.lstm(comm_fea.unsqueeze(0))  # 添加时间步维度
        # lstm_out=lstm_out[:, -1, :]

        # temp = lstm_out
        # lstm_out=self.decoder(tgt=lstm_out,memory=lstm_out)
        # lstm_out = temp+lstm_out

        output = self.output_fc(comm_fea)
        return self.sigmod(output)