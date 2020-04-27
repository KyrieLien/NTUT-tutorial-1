import os

from numpy import concatenate, load, save
from pandas import read_excel
from tqdm import tqdm

from config import Params
from preprocess import Embedding, Segmentation
from torch.utils.data import DataLoader
from model import LoadingDataset, PadSequence, TrainTest


def get_data():
    # 2017 training: 15995 > 15438
    # 2018 testing: 5883 > 5434

    # drop empty values in the same time
    train = read_excel('./data/dataset.xlsx', sheet_name='2017').dropna()
    test = read_excel('./data/dataset.xlsx', sheet_name='2018').dropna()

    # drop unused columns
    train.pop('id')
    train.pop('date')
    test.pop('id')
    test.pop('date')

    # convert to np array
    return train.values, test.values


def main():
    "Prepare data"
    if not os.path.exists(Params.PATH_TRAIN) or not os.path.exists(Params.PATH_TEST):
        # step 1: get data
        train, test = get_data()

        # step 2: text preprocess
        seg = Segmentation()

        seged_train = [[seg.execute(text), label]
                       for text, label in tqdm(train)]
        seged_test = [[seg.execute(text), label] for text, label in tqdm(test)]

        save(Params.PATH_TRAIN, seged_train)
        save(Params.PATH_TEST, seged_test)

    seged_train = load(Params.PATH_TRAIN, allow_pickle=True)
    seged_test = load(Params.PATH_TEST, allow_pickle=True)

    "Word embedding"
    wb_model = Embedding(Params.PATH_WB)

    if not os.path.exists(Params.PATH_WB):
        wb_model.train_vec(concatenate([seged_train[:, 0], seged_test[:, 0]]))

    vec_train = wb_model.infer_vec(seged_train)
    vec_test = wb_model.infer_vec(seged_test)

    "classifier"

    trainingset = LoadingDataset(vec_train)
    testingset = LoadingDataset(vec_test)

    training_loader = DataLoader(
        trainingset, batch_size=Params.B_SIZE, shuffle=True, collate_fn=PadSequence())
    testing_loader = DataLoader(
        testingset, batch_size=Params.B_SIZE, shuffle=True, collate_fn=PadSequence())

    model = TrainTest()

    # --train--
    for i in range(Params.EPOCH):
        print(f'Epoch: {i+1}')
        for X, y, length in tqdm(training_loader):
            model.train_test(X, length, y=y)

        # --test--
        correct = 0
        total = 0
        for X, y, length in tqdm(testing_loader):
            y_pred = model.train_test(X, length)
            total += y.size(0)
            correct += (y_pred == y).sum().item()

        print(f'Acc.: {correct / total}')


if __name__ == "__main__":
    main()
