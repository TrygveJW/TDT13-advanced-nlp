import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import CamembertModel, CamembertTokenizerFast
import numpy as np

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base")
# tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [int(stars) -1 for stars in df['stars']]
        self.texts = [tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['review_body']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained("cmarkea/distilcamembert-base") # 103 params
        # self.bert = CamembertModel.from_pretrained('camembert-base')

        # print(len(list(self.bert.named_parameters())))

        # for name, param in list(self.bert.named_parameters()):
        #     print(name)
        # exit()
        # for name, param in list(self.bert.named_parameters())[:-50]:
            # print('I will be frozen: {}'.format(name))
            # param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_model(path: str):
        model = BertClassifier()
        model.load_state_dict(torch.load(path))
        model.eval()
        return model

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=10)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
        print("using cuda")

        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)


            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += float(batch_loss.item())

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += float(acc)

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # del train_label
            # del mask
            # del input_id
            # del output
            # del batch_loss
            # del acc


        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += float(batch_loss.item())

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += float(acc)

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
