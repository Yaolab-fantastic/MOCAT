import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn import preprocessing, metrics
from models import *
from sklearn.metrics import f1_score
import os
import warnings
warnings.filterwarnings('ignore')
def prepare_data(data_dir, batch_size):

    train_O2 = pd.read_csv(os.path.join(data_dir, '1_tr.csv')).iloc[:, :].to_numpy()
    test_O2 = pd.read_csv(os.path.join(data_dir, '1_te.csv')).iloc[:, :].to_numpy()
    train_O1 = pd.read_csv(os.path.join(data_dir, '2_tr.csv')).iloc[:, :].to_numpy()
    test_O1 = pd.read_csv(os.path.join(data_dir, '2_te.csv')).iloc[:, :].to_numpy()
    train_O3 = pd.read_csv(os.path.join(data_dir, '3_tr.csv')).iloc[:, :].to_numpy()
    test_O3 = pd.read_csv(os.path.join(data_dir, '3_te.csv')).iloc[:, :].to_numpy()
    train_label = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv')).iloc[:, ].to_numpy()
    test_label = pd.read_csv(os.path.join(data_dir, 'labels_te.csv')).iloc[:, ].to_numpy()

    ord_enc = preprocessing.OrdinalEncoder(dtype='int64')
    ord_enc.fit(train_label.reshape(-1, 1))
    y_train = torch.tensor(ord_enc.transform(train_label.reshape(-1, 1))).squeeze()
    y_test = torch.tensor(ord_enc.transform(test_label.reshape(-1, 1))).squeeze()
    print('Classes: ', ord_enc.categories_[0])

    if len(ord_enc.categories_[0]) == 2:  # for reproduce results purpose
        label_weight = torch.tensor([1, 1], dtype=torch.float32)  # 1.5, 1.5, 0.5, 5, 1.5
    else:
        count_label = y_train.unique(return_counts=True)[1].float()
        if count_label.max() / count_label.min() >= 2:
            label_weight = count_label.sum() / count_label / 5  # balanced
        else:
            label_weight = torch.ones_like(count_label)
    print('Weight for these classes:', label_weight)


    X_train_O1_AE = train_O1
    X_train_O2_AE = train_O2
    X_train_O3_AE = train_O3

    scaler_O1 = preprocessing.StandardScaler().fit(X_train_O1_AE)
    scaler_O2 = preprocessing.StandardScaler().fit(X_train_O2_AE)
    scaler_O3 = preprocessing.StandardScaler().fit(X_train_O3_AE)

    X_train_O1_AE = torch.tensor(scaler_O1.transform(X_train_O1_AE), dtype=torch.float32)
    X_train_O1_clf = torch.tensor(scaler_O1.transform(train_O1), dtype=torch.float32)
    X_test_O1 = torch.tensor(scaler_O1.transform(test_O1), dtype=torch.float32)
    X_train_O2_AE = torch.tensor(scaler_O2.transform(X_train_O2_AE), dtype=torch.float32)
    X_train_O2_clf = torch.tensor(scaler_O2.transform(train_O2), dtype=torch.float32)
    X_test_O2 = torch.tensor(scaler_O2.transform(test_O2), dtype=torch.float32)
    X_train_O3_AE = torch.tensor(scaler_O3.transform(X_train_O3_AE), dtype=torch.float32)
    X_train_O3_clf = torch.tensor(scaler_O3.transform(train_O3), dtype=torch.float32)
    X_test_O3 = torch.tensor(scaler_O3.transform(test_O3), dtype=torch.float32)

    train_O1_AE_ds = TensorDataset(X_train_O1_AE)
    train_O2_AE_ds = TensorDataset(X_train_O2_AE)
    train_O3_AE_ds = TensorDataset(X_train_O3_AE)
    train_clf_ds = TensorDataset(X_train_O1_clf, X_train_O2_clf, X_train_O3_clf, y_train)
    test_O1_ds = TensorDataset(X_test_O1)
    test_O2_ds = TensorDataset(X_test_O2)
    test_O3_ds = TensorDataset(X_test_O3)
    test_clf_ds = TensorDataset(X_test_O1, X_test_O2, X_test_O3, y_test)

    if 'BRCA' in data_dir:
        batch_size_clf = batch_size * 2
    elif 'CRC' in data_dir:
        batch_size_clf = batch_size = int(batch_size / 2)
    else:
        batch_size_clf = batch_size
    train_clf_dl = DataLoader(train_clf_ds, batch_size=batch_size_clf, shuffle=True)
    train_O1_AE_dl = DataLoader(train_O1_AE_ds, batch_size=batch_size, shuffle=True)
    train_O2_AE_dl = DataLoader(train_O2_AE_ds, batch_size=batch_size, shuffle=True)
    train_O3_AE_dl = DataLoader(train_O3_AE_ds, batch_size=batch_size, shuffle=True)

    return (train_clf_dl, train_O1_AE_dl, train_O2_AE_dl, train_O3_AE_dl), (
       test_clf_ds, test_O1_ds, test_O2_ds, test_O3_ds), label_weight, ord_enc.categories_[0]


def evaluate(model, testdata, idx2class, result_dir):
    model.eval()
    with torch.no_grad():
        yb = testdata[:][-1]
        preds, f_connect, f_out, tcp_confidence = model(testdata[:][0], testdata[:][1],
                                                        testdata[:][2])
        preds = F.softmax(preds, dim=1)
        _, preds_label = torch.max(preds.data, dim=-1)
        test_acc = (preds_label == yb.data).sum().item() / yb.size(0)

    preds = preds.cpu()
    preds_label = preds_label.data.cpu()
    if len(idx2class) == 2:
        print('Test ACC:',test_acc)
        print('Test AUC:', metrics.roc_auc_score(testdata[:][-1], preds.data[:, 1]))
        print('Test F1:',  f1_score(testdata[:][-1], preds_label))
    clf_report = metrics.classification_report(testdata[:][-1],
                                               preds_label,
                                               target_names=idx2class,
                                               digits=4,
                                               zero_division=0,
                                               output_dict=True)
    clf_df = pd.DataFrame(clf_report)
    clf_df.loc[['precision', 'recall'], 'accuracy'] = np.nan
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(13)
    metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(testdata[:][-1], preds_label),
                                   display_labels=idx2class).plot(cmap='Blues', ax=ax1)
    sns.heatmap(clf_df.iloc[:-1, :].T, annot=True, cmap='Blues', robust=True, ax=ax2, fmt='.2%')

    plt.savefig(os.path.join(result_dir, 'test_results.png'))
    return f1_score(testdata[:][-1], preds_label)

def train_test(data_dir, result_dir, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)


    dev = "cpu"
    device = torch.device(dev)

    print('Loading data...')
    loader, dataset, label_weight, idx2class = prepare_data(data_dir, batch_size)
    train_clf_dl, train_O1_AE_dl, train_O2_AE_dl, train_O3_AE_dl = loader
    test_clf_ds, test_O1_ds, test_O2_ds, test_O3_ds = dataset

    # 定义并保存 O1AE 模型
    # 定义 O1AE 模型并加载参数
    O1AE = O1autoencoder(len(test_clf_ds[0][0]), 2)
    O1AE.load_state_dict(torch.load("O1AE.pt"))
    O1AE.to(device)

    # 打印信息

    # 定义 O2AE 模型并加载参数
    O2AE = O2autoencoder(len(test_clf_ds[0][1]), 2)
    O2AE.load_state_dict(torch.load("O2AE.pt"))
    O2AE.to(device)

    # 打印信息

    # 定义 O3AE 模型并加载参数
    O3AE = O3autoencoder(len(test_clf_ds[0][2]), 2)
    O3AE.load_state_dict(torch.load("O3AE.pt"))
    O3AE.to(device)

    # 打印信息

    # 定义 fusion model 模型并加载参数
    clf = Subtyping_model(O1_encoder=O1AE.encoder,
                          O2_encoder=O2AE.encoder,
                          O3_encoder=O3AE.encoder,
                          subtypes=len(label_weight))
    state_dict = torch.load("clf.pt", map_location=torch.device('cpu'))

    # Load the state dictionary into the model
    clf.load_state_dict(state_dict)
    clf.to(device)

    # 打印 clf_test_acc

    # 评估模型
    evaluate(clf, test_clf_ds, idx2class, result_dir)



