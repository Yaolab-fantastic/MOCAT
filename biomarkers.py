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
# 读取三个分组数据的训练集和测试集
# 标签数据的读取方式将在后面进行讲解
data_dir = 'ROSMAP'
result_dir = 'ROSMAP'
lr_O1AE = 1e-4
n_epoch_O1AE = 10
lr_O2AE = 1e-4
n_epoch_O2AE = 10
lr_AE = 1e-6
lr_clf = 2e-6
patience = 20
batch_size = 36
wd_AE = 0.0000
test_F1 = []
if 'BRCA' in data_dir:
    wd_clf = 5e-1
else:
    wd_clf = 3e-1


def Feature_ablation():
    featnames_1 = pd.read_csv(os.path.join(data_dir, "1_featname.csv"), header=None).T
    featnames_2 = pd.read_csv(os.path.join(data_dir, "2_featname.csv"), header=None).T
    featnames_3 = pd.read_csv(os.path.join(data_dir, "3_featname.csv"), header=None).T
    feature_all = pd.concat((featnames_1, featnames_2, featnames_3)).values.flatten().tolist()
    print(len(feature_all))
    for i in range(3):
        feature_study(data_dir, omic=i + 1)
    print("Feature ablation F1 shape:", len(test_F1))
    print("Feature ablation F1:", test_F1)
    pairs = [(num, idx) for idx, num in enumerate(test_F1)]
    sorted_pairs = sorted(pairs, key=lambda x: x[0])[:50]
    indices = [pair[1] for pair in sorted_pairs]
    print("positions:", indices)

    data = []
    data1 = []
    data2 = []
    data3 = []
    for pos in indices:
        data.append(feature_all[pos])
    print('all biomarkers:',data)
    mRNA_list = [i for i in range(len(indices)) if indices[i] < 200]
    DNA_list = [i for i in range(len(indices)) if 200 < indices[i] < 400]
    miRNA_list = [i for i in range(len(indices)) if 400 < indices[i] < 600]
    for pos1 in mRNA_list:
        data1.append(data[pos1])
    print('mRNA:',data1)
    for pos2 in DNA_list:
        data2.append(data[pos2])
    print('DNA:',data2)
    for pos3 in miRNA_list:
        data3.append(data[pos3])
    print('miRNA:',data3)
def feature_study(data_dir,  omic):
    train_O2 = pd.read_csv(os.path.join(data_dir, '1_tr.csv'))
    test_O2 = pd.read_csv(os.path.join(data_dir, '1_te.csv'))
    train_O1 = pd.read_csv(os.path.join(data_dir, '2_tr.csv'))
    test_O1 = pd.read_csv(os.path.join(data_dir, '2_te.csv'))
    train_O3 = pd.read_csv(os.path.join(data_dir, '3_tr.csv'))
    test_O3 = pd.read_csv(os.path.join(data_dir, '3_te.csv'))
    featnames_1 = pd.read_csv(os.path.join(data_dir, "1_featname.csv"), header=None).T
    featnames_2 = pd.read_csv(os.path.join(data_dir, "2_featname.csv"), header=None).T
    featnames_3 = pd.read_csv(os.path.join(data_dir, "3_featname.csv"), header=None).T
    feature_all = np.hstack((featnames_1, featnames_2, featnames_3))
    data_1_tr = np.vstack((featnames_1, train_O2))
    data_1_te = np.vstack((featnames_1, test_O2))
    data_2_tr = np.vstack((featnames_2, train_O1))
    data_2_te = np.vstack((featnames_2, test_O1))
    data_3_tr = np.vstack((featnames_3, train_O3))
    data_3_te = np.vstack((featnames_3, test_O3))
    data_1_tr = pd.DataFrame(data_1_tr)
    data_1_te = pd.DataFrame(data_1_te)
    data_2_tr = pd.DataFrame(data_2_tr)
    data_2_te = pd.DataFrame(data_2_te)
    data_3_tr = pd.DataFrame(data_3_tr)
    data_3_te = pd.DataFrame(data_3_te)
    if omic == 1:
        for featname in featnames_1:
            # 去除当前特征
            data_1_tr_no_drop = data_1_tr.copy()
            data_1_te_no_drop = data_1_te.copy()
            data_2_tr_no_drop = data_2_tr.copy()
            data_2_te_no_drop = data_2_te.copy()
            data_3_tr_no_drop = data_3_tr.copy()
            data_3_te_no_drop = data_3_te.copy()
            data_1_tr_no_drop.loc[:, featname] = 0.0
            data_1_te_no_drop.loc[:, featname] = 0.0
            train_O2 = data_1_tr_no_drop.iloc[1:, :].to_numpy()
            test_O2 = data_1_te_no_drop.iloc[1:, :].to_numpy()
            train_O1 = data_2_tr_no_drop.iloc[1:, :].to_numpy()
            test_O1 = data_2_te_no_drop.iloc[1:, :].to_numpy()
            train_O3 = data_3_tr_no_drop.iloc[1:, :].to_numpy()
            test_O3 = data_3_te_no_drop.iloc[1:, :].to_numpy()
            train_label = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv')).iloc[:, ].to_numpy()
            test_label = pd.read_csv(os.path.join(data_dir, 'labels_te.csv')).iloc[:, ].to_numpy()
            ord_enc = preprocessing.OrdinalEncoder(dtype='int64')
            ord_enc.fit(train_label.reshape(-1, 1))
            y_test = torch.tensor(ord_enc.transform(test_label.reshape(-1, 1))).squeeze()

            scaler_O1 = preprocessing.StandardScaler().fit(train_O1)
            scaler_O2 = preprocessing.StandardScaler().fit(train_O2)
            scaler_O3 = preprocessing.StandardScaler().fit(train_O3)

            test_O1 = torch.tensor(scaler_O1.transform(test_O1), dtype=torch.float32)
            test_O2 = torch.tensor(scaler_O2.transform(test_O2), dtype=torch.float32)
            test_O3 = torch.tensor(scaler_O3.transform(test_O3), dtype=torch.float32)
            test_clf_ds = TensorDataset(test_O1, test_O2, test_O3, y_test)
            if len(ord_enc.categories_[0]) == 2:  # for reproduce results purpose
                label_weight = torch.tensor([1, 1], dtype=torch.float32)  # 1.5, 1.5, 0.5, 5, 1.5
            else:
                count_label = y_test.unique(return_counts=True)[1].float()
                if count_label.max() / count_label.min() >= 2:
                    label_weight = count_label.sum() / count_label / 5  # balanced
                else:
                    label_weight = torch.ones_like(count_label)

            dev = "cpu"
            device = torch.device(dev)
            state_dict_O1AE = torch.load("O1AE.pt")
            O1AE = O1autoencoder(len(test_clf_ds[0][0]), 2)
            O1AE.load_state_dict(state_dict_O1AE)
            O1AE.to(device)
            state_dict_O2AE = torch.load("O2AE.pt")
            O2AE = O2autoencoder(len(test_clf_ds[0][1]), 2)
            O2AE.load_state_dict(state_dict_O2AE)
            O2AE.to(device)
            state_dict_O3AE = torch.load("O3AE.pt")
            O3AE = O3autoencoder(len(test_clf_ds[0][2]), 2)
            O3AE.load_state_dict(state_dict_O3AE)
            O3AE.to(device)
            state_dict_clf = torch.load("clf.pt", map_location=torch.device('cpu'))
            clf = Subtyping_model(O1_encoder=O1AE.encoder,
                                  O2_encoder=O2AE.encoder,
                                  O3_encoder=O3AE.encoder,
                                  subtypes=len(label_weight))
            clf.to(device)
            clf.load_state_dict(state_dict_clf)

            F1 = evaluate(clf, test_clf_ds, ord_enc.categories_[0], result_dir,featname,omic)
            test_F1.append(F1)
    if omic == 2:
        for featname in featnames_2:
            # 去除当前特征
            data_1_tr_no_drop = data_1_tr.copy()
            data_1_te_no_drop = data_1_te.copy()
            data_2_tr_no_drop = data_2_tr.copy()
            data_2_te_no_drop = data_2_te.copy()
            data_3_tr_no_drop = data_3_tr.copy()
            data_3_te_no_drop = data_3_te.copy()
            data_2_tr_no_drop.loc[:, featname] = 0.0
            data_2_te_no_drop.loc[:, featname] = 0.0
            train_O2 = data_1_tr_no_drop.iloc[1:, :].to_numpy()
            test_O2 = data_1_te_no_drop.iloc[1:, :].to_numpy()
            train_O1 = data_2_tr_no_drop.iloc[1:, :].to_numpy()
            test_O1 = data_2_te_no_drop.iloc[1:, :].to_numpy()
            train_O3 = data_3_tr_no_drop.iloc[1:, :].to_numpy()
            test_O3 = data_3_te_no_drop.iloc[1:, :].to_numpy()
            train_label = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv')).iloc[:, ].to_numpy()
            test_label = pd.read_csv(os.path.join(data_dir, 'labels_te.csv')).iloc[:, ].to_numpy()
            ord_enc = preprocessing.OrdinalEncoder(dtype='int64')
            ord_enc.fit(train_label.reshape(-1, 1))
            y_test = torch.tensor(ord_enc.transform(test_label.reshape(-1, 1))).squeeze()

            scaler_O1 = preprocessing.StandardScaler().fit(train_O1)
            scaler_O2 = preprocessing.StandardScaler().fit(train_O2)
            scaler_O3 = preprocessing.StandardScaler().fit(train_O3)

            test_O1 = torch.tensor(scaler_O1.transform(test_O1), dtype=torch.float32)
            test_O2 = torch.tensor(scaler_O2.transform(test_O2), dtype=torch.float32)
            test_O3 = torch.tensor(scaler_O3.transform(test_O3), dtype=torch.float32)
            test_clf_ds = TensorDataset(test_O1, test_O2, test_O3, y_test)
            if len(ord_enc.categories_[0]) == 2:  # for reproduce results purpose
                label_weight = torch.tensor([1, 1], dtype=torch.float32)  # 1.5, 1.5, 0.5, 5, 1.5
            else:
                count_label = y_test.unique(return_counts=True)[1].float()
                if count_label.max() / count_label.min() >= 2:
                    label_weight = count_label.sum() / count_label / 5  # balanced
                else:
                    label_weight = torch.ones_like(count_label)


            dev = "cpu"
            device = torch.device(dev)
            state_dict_O1AE = torch.load("O1AE.pt")
            O1AE = O1autoencoder(len(test_clf_ds[0][0]), 2)
            O1AE.load_state_dict(state_dict_O1AE)
            O1AE.to(device)
            state_dict_O2AE = torch.load("O2AE.pt")
            O2AE = O2autoencoder(len(test_clf_ds[0][1]), 2)
            O2AE.load_state_dict(state_dict_O2AE)
            O2AE.to(device)
            state_dict_O3AE = torch.load("O3AE.pt")
            O3AE = O3autoencoder(len(test_clf_ds[0][2]), 2)
            O3AE.load_state_dict(state_dict_O3AE)
            O3AE.to(device)
            state_dict_clf = torch.load("clf.pt", map_location=torch.device('cpu'))
            clf = Subtyping_model(O1_encoder=O1AE.encoder,
                                  O2_encoder=O2AE.encoder,
                                  O3_encoder=O3AE.encoder,
                                  subtypes=len(label_weight))
            clf.to(device)
            clf.load_state_dict(state_dict_clf)

            F1 = evaluate(clf, test_clf_ds, ord_enc.categories_[0], result_dir,featname,omic)
            test_F1.append(F1)
    if omic == 3:
        for featname in featnames_3:
            # 去除当前特征
            data_1_tr_no_drop = data_1_tr.copy()
            data_1_te_no_drop = data_1_te.copy()
            data_2_tr_no_drop = data_2_tr.copy()
            data_2_te_no_drop = data_2_te.copy()
            data_3_tr_no_drop = data_3_tr.copy()
            data_3_te_no_drop = data_3_te.copy()
            data_3_tr_no_drop.loc[:, featname] = 0.0
            data_3_te_no_drop.loc[:, featname] = 0.0
            train_O2 = data_1_tr_no_drop.iloc[1:, :].to_numpy()
            test_O2 = data_1_te_no_drop.iloc[1:, :].to_numpy()
            train_O1 = data_2_tr_no_drop.iloc[1:, :].to_numpy()
            test_O1 = data_2_te_no_drop.iloc[1:, :].to_numpy()
            train_O3 = data_3_tr_no_drop.iloc[1:, :].to_numpy()
            test_O3 = data_3_te_no_drop.iloc[1:, :].to_numpy()
            train_label = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv')).iloc[:, ].to_numpy()
            test_label = pd.read_csv(os.path.join(data_dir, 'labels_te.csv')).iloc[:, ].to_numpy()
            ord_enc = preprocessing.OrdinalEncoder(dtype='int64')
            ord_enc.fit(train_label.reshape(-1, 1))
            y_test = torch.tensor(ord_enc.transform(test_label.reshape(-1, 1))).squeeze()

            scaler_O1 = preprocessing.StandardScaler().fit(train_O1)
            scaler_O2 = preprocessing.StandardScaler().fit(train_O2)
            scaler_O3 = preprocessing.StandardScaler().fit(train_O3)

            test_O1 = torch.tensor(scaler_O1.transform(test_O1), dtype=torch.float32)
            test_O2 = torch.tensor(scaler_O2.transform(test_O2), dtype=torch.float32)
            test_O3 = torch.tensor(scaler_O3.transform(test_O3), dtype=torch.float32)
            test_clf_ds = TensorDataset(test_O1, test_O2, test_O3, y_test)
            if len(ord_enc.categories_[0]) == 2:  # for reproduce results purpose
                label_weight = torch.tensor([1, 1], dtype=torch.float32)  # 1.5, 1.5, 0.5, 5, 1.5
            else:
                count_label = y_test.unique(return_counts=True)[1].float()
                if count_label.max() / count_label.min() >= 2:
                    label_weight = count_label.sum() / count_label / 5  # balanced
                else:
                    label_weight = torch.ones_like(count_label)


            dev = "cpu"
            device = torch.device(dev)
            state_dict_O1AE = torch.load("O1AE.pt")
            O1AE = O1autoencoder(len(test_clf_ds[0][0]), 2)
            O1AE.load_state_dict(state_dict_O1AE)
            O1AE.to(device)
            state_dict_O2AE = torch.load("O2AE.pt")
            O2AE = O2autoencoder(len(test_clf_ds[0][1]), 2)
            O2AE.load_state_dict(state_dict_O2AE)
            O2AE.to(device)
            state_dict_O3AE = torch.load("O3AE.pt")
            O3AE = O3autoencoder(len(test_clf_ds[0][2]), 2)
            O3AE.load_state_dict(state_dict_O3AE)
            O3AE.to(device)
            state_dict_clf = torch.load("clf.pt", map_location=torch.device('cpu'))
            clf = Subtyping_model(O1_encoder=O1AE.encoder,
                                  O2_encoder=O2AE.encoder,
                                  O3_encoder=O3AE.encoder,
                                  subtypes=len(label_weight))
            clf.to(device)
            clf.load_state_dict(state_dict_clf)

            F1 = evaluate(clf, test_clf_ds, ord_enc.categories_[0], result_dir,featname,omic)
            test_F1.append(F1)

def evaluate(model, testdata, idx2class, result_dir,featname,omic):
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
        print('omic',omic,'ablated feature:',featname,'Test F1:',  f1_score(testdata[:][-1], preds_label))
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
Feature_ablation()
