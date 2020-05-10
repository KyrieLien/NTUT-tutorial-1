import torch
from torch import nn, tensor
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.optim.adam import Adam
from torch.utils.data import Dataset

from config import Params

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PadSequence:

    def __call__(self, batch):
        """Padding to max length when batch

        Args:
            batch (tensor): from dataloader

        Returns:
            sequence_padded (tensor): padded X
            label (tensor): y
            length (tensor): original X length
        """
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sequences = []
        labels = []
        lengths = []

        for X, y in sorted_batch:
            sequences.append(tensor(X))
            labels.append(y)
            lengths.append(len(X))
        sequence_padded = pad_sequence(sequences, batch_first=True)
        return sequence_padded.to(DEVICE), tensor(labels).long().to(DEVICE), tensor(lengths).to(DEVICE)


class LoadingDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        X, y = self.dataset[index]
        return X, y

    def __len__(self):
        return len(self.dataset)


class _Classifier(nn.Module):
    """
    Bidirectional GRU
    """

    def __init__(self):
        """
        build your neural network, contain GRU and linear
        """
        super().__init__()
        self.gru = nn.GRU(Params.INPUT_SIZE, Params.HIDDEN_SIZE,
                          batch_first=True, bidirectional=True)
        # edit by yourself
        self.fc_out = nn.Sequential(
            nn.Linear(200, 64),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(64, 2)
        )

    def forward(self, X, X_len):
        """
        feeding X and length into your model

        Args:
            X (tensor): training data
            X_len (tensor): length

        Returns:
            output (tensor): output hidden
        """
        # remove padding value for recover data
        X_pack = pack_padded_sequence(X, X_len, batch_first=True)
        # into model
        _, h_s = self.gru(X_pack)
        # bidirection contain two of neural, forward and backward, [2, batch size, hidden size](e.g. [2, 128, 100])
        # so we want to split them and concat together
        forward, backward = torch.chunk(h_s, 2, 0)
        # [2, 128, 100] > [1, 128, 200] > [128, 200]
        # into fully connected
        H = torch.cat([forward, backward], dim=-1).squeeze()
        output = self.fc_out(H)

        return output


class TrainTest:
    """
    initialize your model and build you optimizer and loss function
    """

    def __init__(self):
        self.model = _Classifier().to(DEVICE)
        self.optimizer = Adam(self.model.parameters(), lr=Params.LR)
        self.criterion = CrossEntropyLoss()

    def train_test(self, X, X_len, **kwargs):
        """
        train and test function

        Args:
            X (tensor): training data
            X_len (tensor): length

        Returns:
            y_pred (int): predict values
        """
        y = kwargs.pop('y', None)

        self.optimizer.zero_grad()

        if y is None:
            # eval can disable dropout
            self.model.eval()
            with torch.no_grad():
                output = self.model(X, X_len)
                y_pred = output.detach().argmax(1)
            return y_pred
        else:
            self.model.train()
            output = self.model(X, X_len)
            loss = self.criterion(output, y)
            # if multi gpu, remember use loss.mean()
            loss.backward()
            self.optimizer.step()

    def save(self):
        torch.save(self.model.state_dict(), Params.PATH_SAVE)
