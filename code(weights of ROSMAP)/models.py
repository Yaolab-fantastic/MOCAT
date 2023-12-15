import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
class O1autoencoder(nn.Module):
    def __init__(self, fan_in, subtypes):
        super(O1autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(fan_in, 3000),
            nn.BatchNorm1d(3000),
            nn.ELU(),
            nn.Linear(3000, 180),
            nn.ELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(180,3000),
            nn.ELU(),
            nn.Linear(3000, fan_in)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(180 , 1024),
            nn.ELU(),
            nn.Linear(1024, subtypes)
        )
    def forward(self, x):
        x = self.encoder(x)
        pre = self.classifier(x)
        x = self.decoder(x)
        return x,pre

class O2autoencoder(nn.Module):
    def __init__(self, fan_in, subtypes):
        super(O2autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(fan_in, 3000),
            nn.BatchNorm1d(3000),
            nn.ELU(),
            nn.Linear(3000, 180),
            nn.ELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(180, 3000),
            nn.ELU(),
            nn.Linear(3000, fan_in)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(180, 1024),
            nn.ELU(),
            nn.Linear(1024, subtypes)
        )

    def forward(self, x):
        x = self.encoder(x)
        pre = self.classifier(x)
        x = self.decoder(x)
        return x,pre

class O3autoencoder(nn.Module):
    def __init__(self, fan_in, subtypes):
        super(O3autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(fan_in, 2048),
            nn.BatchNorm1d(2048),
            nn.ELU(),
            nn.Linear(2048, 180),
            nn.ELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(180, 2048),
            nn.ELU(),
            nn.Linear(2048, fan_in)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(180, 512),
            nn.ELU(),
            nn.Linear(512, subtypes)
        )

    def forward(self, x):
        x = self.encoder(x)
        pre = self.classifier(x)
        x = self.decoder(x)
        return x,pre

class Subtyping_model(nn.Module):
    def __init__(self, O1_encoder, O2_encoder,O3_encoder, subtypes):
        super(Subtyping_model, self).__init__()

        self.O1_repr = nn.Sequential(*list(O1_encoder.children())[1:])
        self.O2_repr = nn.Sequential(*list(O2_encoder.children())[1:])
        self.O3_repr = nn.Sequential(*list(O3_encoder.children())[1:])
        self.classifier_encoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(180+180+180, 3000),
            nn.Dropout(0.5),
            nn.ELU(),
            nn.Linear(3000, 2048),
            nn.Dropout(0.5),
            nn.ELU(),
            nn.Linear(2048,500)
        )
        self.classifier_decoder = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(500, 2048),
            nn.Dropout(0.5),
            nn.ELU(),
            nn.Linear(2048,3000),
            nn.Dropout(0.5),
            nn.ELU(),
            nn.Linear(3000, 180+180+180)
        )
        self.multihead_attn = MultiheadAttention(500, 4)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(500, 2800),
            nn.Dropout(0.5),
            nn.ELU(),
            nn.Linear(2800,1024),
            nn.Dropout(0.15),
            nn.ELU(),
            nn.Linear(1024, subtypes)
        )
        self.tcp = nn.Sequential(
            nn.Linear(500, 1)
        )

    def forward(self, x1, x2, x3):
        O1_ft = self.O1_repr(x1)
        O2_ft = self.O2_repr(x2)
        O3_ft = self.O3_repr(x3)
        feature_connect = torch.hstack((O1_ft, O2_ft, O3_ft))
        classifier_ae_feature = self.classifier_encoder(torch.hstack((O1_ft, O2_ft, O3_ft)))
        classifier_ae_out = self.classifier_decoder(classifier_ae_feature)
        x = classifier_ae_feature

        attn_output, attn_weights = self.multihead_attn(x, x, x)
        x = x + attn_output
        tcp_confidence = self.tcp(x)
        tcp_confidence = torch.sigmoid(tcp_confidence)
        pred = self.classifier(x)
        return pred, feature_connect, classifier_ae_out, tcp_confidence


