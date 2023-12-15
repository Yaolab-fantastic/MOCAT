
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing, metrics
from models import *
import os
from utils import evaluate,train_test
data_dir = '/data/xiaohan/MGN/ROSMAP'
result_dir = '/data/xiaohan/png/MGN/ROSMAP'
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
    train_test(data_dir, result_dir, batch_size)
    for i in range(3):
        feature_study(data_dir, omic=i + 1)
    print("Feature ablation F1 shape:", len(test_F1))
    print("Feature ablation F1:", test_F1)
    pairs = [(num, idx) for idx, num in enumerate(test_F1)]
    sorted_pairs = sorted(pairs, key=lambda x: x[0])[:30]
    indices = [pair[1] for pair in sorted_pairs]
    print("positions:", indices)
    data = []
    for pos in indices:
        data.append(feature_all[pos])
    print(data)

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

            if torch.cuda.is_available():
                dev = "cuda:0"
            else:
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
            state_dict_clf = torch.load("clf.pt")
            clf = Subtyping_model(O1_encoder=O1AE.encoder,
                                  O2_encoder=O2AE.encoder,
                                  O3_encoder=O3AE.encoder,
                                  subtypes=len(label_weight))
            clf.to(device)
            clf.load_state_dict(state_dict_clf)

            F1 = evaluate(clf, test_clf_ds, ord_enc.categories_[0], result_dir)
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

            if torch.cuda.is_available():
                dev = "cuda:0"
            else:
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
            state_dict_clf = torch.load("clf.pt")
            clf = Subtyping_model(O1_encoder=O1AE.encoder,
                                  O2_encoder=O2AE.encoder,
                                  O3_encoder=O3AE.encoder,
                                  subtypes=len(label_weight))
            clf.to(device)
            clf.load_state_dict(state_dict_clf)

            F1 = evaluate(clf, test_clf_ds, ord_enc.categories_[0], result_dir)
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

            if torch.cuda.is_available():
                dev = "cuda:0"
            else:
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
            state_dict_clf = torch.load("clf.pt")
            clf = Subtyping_model(O1_encoder=O1AE.encoder,
                                  O2_encoder=O2AE.encoder,
                                  O3_encoder=O3AE.encoder,
                                  subtypes=len(label_weight))
            clf.to(device)
            clf.load_state_dict(state_dict_clf)

            F1 = evaluate(clf, test_clf_ds, ord_enc.categories_[0], result_dir)
            test_F1.append(F1)


Feature_ablation()

