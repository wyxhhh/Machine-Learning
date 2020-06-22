import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.fc = nn.Sequential(
            nn.Linear(2048, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.proj = nn.Linear(self.L*self.K, 10)

    def forward(self, x):
        x = x.squeeze(0)
        x = x.squeeze(1)
        
        H = self.fc(x)
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        out = self.proj(M)

        return torch.sigmoid(out)

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        # Y = Y.float()
        Y_prob = self.forward(X)[0]
        Y = Y[0]
        Y_hat = []
        # p = np.sum(np.array(Y))
        # print(p)
        for i in range(len(Y_prob)):
            Y_hat.append(torch.ge(Y_prob[i], 0.5).float())
        error = 0
        for i in range(len(Y_hat)):
            if int(Y_hat[i]) != int(Y[i]):
                error += 1
                break
        #     error += 1. - Y_hat[i].eq(Y[i].float).cpu().float().mean().item()

        return error, Y_prob, Y_hat

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        Y = Y.float()
        loss = nn.BCELoss(reduction ='sum')
        Y_prob = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # print(Y_prob)
        # print(Y)
        output = loss(Y_prob, Y)
        print(output.item())
        return output

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.fc = nn.Sequential(
            nn.Linear(2048, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.proj = nn.Linear(self.L*self.K, 10)

    def forward(self, x):
        x = x.squeeze(0)
        x = x.squeeze(1)
        
        H = self.fc(x)

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        out = self.proj(M)
        return torch.sigmoid(out)


    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        # Y = Y.float()
        Y_prob = self.forward(X)[0]
        Y = Y[0]
        Y_hat = []
        # p = np.sum(np.array(Y))
        # print(p)
        for i in range(len(Y_prob)):
            Y_hat.append(torch.ge(Y_prob[i], 0.5).float())
        error = 0
        for i in range(len(Y_hat)):
            if int(Y_hat[i]) != int(Y[i]):
                error += 1
                break
        #     error += 1. - Y_hat[i].eq(Y[i].float).cpu().float().mean().item()

        return error, Y_prob, Y_hat

    def calculate_objective(self, X, Y):
        # Y = Y.float()
        Y = Y.float()
        loss = nn.BCELoss(reduction ='sum')
        Y_prob = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # print(Y_prob)
        # print(Y)
        output = loss(Y_prob, Y)
        print(output.item())
        return output
