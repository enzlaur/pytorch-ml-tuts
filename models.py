# ==== IMPORTS ====
import torch
from torch import nn
# ---- TEST MODEL ----
kernel_size = 16
padding_size= int( (kernel_size/2) ) # If odd, add -1

class ae_sampler(nn.Module):
    def __init__(self):
        super(ae_sampler, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 40, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(40),
            nn.Conv1d(40, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.Conv1d(20, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.Conv1d(20, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.Conv1d(20, 40, kernel_size, stride=2, padding=8 ), nn.ELU(True), # 521
            nn.BatchNorm1d(40),
            nn.Conv1d(40, 1, kernel_size, stride=1, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 40, kernel_size, stride=1, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(40),
            nn.ConvTranspose1d(40, 20, kernel_size, stride=2, padding=8 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.ConvTranspose1d(20, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.ConvTranspose1d(20, 20, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(20),
            nn.ConvTranspose1d(20, 40, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(40),
            nn.ConvTranspose1d(40, 1, kernel_size, stride=2, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(1),
            nn.ConvTranspose1d(1, 1, kernel_size, stride=1, padding=8 ), nn.ELU(True), # 521
            nn.BatchNorm1d(1),
            nn.ConvTranspose1d(1, 1, kernel_size, stride=1, padding=7 ), nn.ELU(True), # 521
            nn.BatchNorm1d(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

class cnn_denoiser(nn.Module):    
    def __init__(self):
        super(cnn_denoiser, self).__init__()
        
        self.denoiser = nn.Sequential(
            nn.Conv1d(1, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(True),
            # nn.AvgPool1d(kernel_size=2, stride=4),

            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(True),
            # nn.AvgPool1d(kernel_size=2, stride=4),

            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(True),
            # nn.AvgPool1d(kernel_size=2, stride=4),

            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(True),
            # nn.AvgPool1d(kernel_size=2, stride=4),

            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(True),
            # nn.AvgPool1d(kernel_size=2, stride=4),

            nn.Conv1d(36, 36, kernel_size=19, stride=1, padding=9),
            nn.BatchNorm1d(36),
            nn.ReLU(True),
            # nn.AvgPool1d(kernel_size=2, stride=4),

            nn.Linear(1024, 1024)
        )

    def forward(self, x):
        x = self.denoiser(x)
        return x
