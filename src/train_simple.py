import datetime
import math
import os.path
import pickle
import time

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizerFast, CamembertTokenizer
import numpy as np

from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base")
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")


camembert = CamembertModel.from_pretrained('camembert-base')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, labels, transformed):

        # self.labels = [int(stars) -1 for stars in df['stars']]
        # tokenized = tokenizer( df['review_body'].tolist(),
        #                        padding='max_length', max_length = 512, truncation=True,
        #                         return_tensors="pt")
        #
        # with torch.no_grad():
        #     self.transformed = camembert(tokenized["input_ids"],tokenized['attention_mask'])
        self.labels = labels
        self.transformed = transformed

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.transformed[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class LinearModel(nn.Module):

    def __init__(self,
                 dropout=0.3):


        # self.bert = AutoModel.from_pretrained("cmarkea/distilcamembert-base") # 103 params
        # self.bert = CamembertModel.from_pretrained('camembert-base')

        # print(len(list(self.bert.named_parameters())))

        # for name, param in list(self.bert.named_parameters()):
        #     print(name)
        # exit()
        # for name, param in list(self.bert.named_parameters())[:-50]:
            # print('I will be frozen: {}'.format(name))
            # param.requires_grad = False
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(768, 200),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(200, 200),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(200, 5),
        #     nn.ReLU(),
        # )


        self.model = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(25, 5),
            nn.ReLU(),
        )

    def forward(self, inputs):
        inputs = inputs.to(torch.float32)
        out = self.model(inputs)


        return out

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path: str):
        model = LinearModel()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

def transform(dat):

    tokenized = tokenizer(dat['review_body'].tolist(),
                          padding='max_length', max_length=512, truncation=True,
                          return_tensors="pt")

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    transformed = torch.zeros([len(input_ids), 768],dtype=torch.float64, device="cuda")

    bs = 100
    batches = math.floor(len(input_ids)/bs)
    with torch.no_grad():
        for n in tqdm(range(batches)):
            from_idx = n * bs
            to_idx = (n+1) * bs

            ids = input_ids[from_idx:to_idx].cuda()
            mask = attention_mask[from_idx:to_idx].cuda()

            v, out = camembert.cuda()(input_ids=ids, attention_mask=mask, return_dict=False)
            transformed[from_idx:to_idx] = out
            del ids
            del mask
            del out
            del v

        if len(tokenized) != batches* bs:
            from_idx = batches * bs
            to_idx = len(input_ids)
            ids = input_ids[from_idx:to_idx].cuda()
            mask = attention_mask[from_idx:to_idx].cuda()
            _, out = camembert.cuda()(input_ids=ids, attention_mask=mask, return_dict=False)
            transformed[from_idx:to_idx] = out

    return transformed

def train_simplefied(model, train_data, val_data, learning_rate, epochs):

    train_fp = "./train_pickle"
    valid_fp = "./valid_pickle"
    if os.path.exists(train_fp):
        train_x = torch.load(train_fp).cpu()
    else:
        train_x = transform(train_data).cpu()
        torch.save(train_x,train_fp)

    if os.path.exists(valid_fp):
        valid_x = torch.load(valid_fp).cpu()
    else:
        valid_x = transform(val_data).cpu()
        torch.save(valid_x, valid_fp)

    train_y =  [int(stars) -1 for stars in train_data['stars']]
    valid_y =  [int(stars) -1 for stars in val_data['stars']]
    # self.labels = [int(stars) -1 for stars in df['stars']]
    # tokenized = tokenizer( train_data['review_body'].tolist(),
    #                        padding='max_length', max_length = 512, truncation=True,
    #                         return_tensors="pt")



    # with torch.no_grad():
    #     self.transformed = camembert(tokenized["input_ids"],tokenized['attention_mask'])
    train, val = Dataset(train_y,train_x), Dataset(valid_y, valid_x)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=10)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    train_loss_list = []
    train_acc_list = []
    train_off_by_1_list = []

    val_loss_list = []
    val_acc_list = []
    val_off_by_1_list = []

    iter_times = []

    if use_cuda:
        print("using cuda")

        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        time_0 = time.time()

        total_acc_train = 0
        total_loss_train = 0
        total_off_by_1_acc_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            model.train()

            train_label = train_label.to(device)
            train_input = train_input.to(device)

            output = model(train_input)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += float(batch_loss.item())


            with torch.no_grad():
                output_mask = torch.ones_like(output, dtype=torch.bool)

                max_idx = output.argmax(dim=1)
                acc = (max_idx == train_label).sum().item()
                total_acc_train += float(acc)

                max_idx_rs = torch.reshape(max_idx, (-1,1))
                output_mask=  torch.scatter(input=output_mask,dim=1, index=max_idx_rs, value=False)
                masked_out = torch.reshape(output[output_mask], (output.shape[0], output.shape[1]-1))

                next_max_idx =  masked_out.argmax(dim=1)

                off_by_1_acc = torch.logical_or(torch.eq(max_idx, train_label), torch.eq(next_max_idx, train_label)).sum().item()
                total_off_by_1_acc_train += off_by_1_acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()


        iter_times.append(time.time()-time_0)

        total_acc_val = 0
        total_loss_val = 0
        total_off_by_1_acc_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                model.eval()

                val_label = val_label.to(device)
                val_input = val_input.to(device)

                output = model(val_input)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += float(batch_loss.item())


                max_idx = output.argmax(dim=1)
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += float(acc)


                output_mask = torch.ones_like(output, dtype=torch.bool)
                max_idx_rs = torch.reshape(max_idx, (-1,1))
                output_mask=  torch.scatter(input=output_mask,dim=1, index=max_idx_rs, value=False)
                masked_out = torch.reshape(output[output_mask], (output.shape[0], output.shape[1]-1))

                next_max_idx =  masked_out.argmax(dim=1)

                off_by_1_acc = torch.logical_or(torch.eq(max_idx, val_label), torch.eq(next_max_idx, val_label)).sum().item()
                total_off_by_1_acc_val += off_by_1_acc


        val_loss_list.append(float(total_loss_val)/len(val_data))
        val_acc_list.append(float(total_acc_val)/len(val_data))
        val_off_by_1_list.append(float(total_off_by_1_acc_val)/len(val_data))

        train_loss_list.append(float(total_loss_train)/len(train_data))
        train_acc_list.append(float(total_acc_train)/len(train_data))
        train_off_by_1_list.append(float(total_off_by_1_acc_train)/len(train_data))

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .5f} \
                | train acc off by 1: {total_off_by_1_acc_train / len(train_data): .5f} \
                | Train Accuracy: {total_acc_train / len(train_data): .5f} \
                | Val Loss: {total_loss_val / len(val_data): .5f} \
                | Val acc off by 1: {total_off_by_1_acc_val / len(val_data): .5f} \
                | Val Accuracy: {total_acc_val / len(val_data): .5f}')

    data_df = pd.DataFrame({
        "val_loss": val_loss_list,
        "val_acc": val_acc_list,
        "val_off_by_1": val_off_by_1_list,

        "train_loss": train_loss_list,
        "train_acc": train_acc_list,
        "train_off_by_1": train_off_by_1_list,

        "epoch_time_s": iter_times,
    })
    data_df.to_csv(f"./train_csv_{datetime.datetime.now()}")