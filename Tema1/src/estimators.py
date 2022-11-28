import torch.nn as nn

class NNEstimator_OLD(nn.Module):

    def __init__(self, action_num: int, input_ch: int = 4, nf: int = 32) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_ch, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Linear(16 * 60 * 60, 16 * 60),
            nn.ReLU(inplace=True),
            nn.Linear(16 * 60, nf),
            nn.ReLU(inplace=True),
            nn.Linear(nf, action_num)
        )

    def forward(self, x):
        x =  self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return x


class NNEstimator(nn.Module):

    def __init__(self, action_num: int, input_ch: int = 4, nf: int = 32) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_ch, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 36, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 36, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.Linear(36 * 58 * 58, nf),
            nn.ReLU(inplace=True),
            nn.Linear(nf, action_num)
        )

    def forward(self, x):
        x =  self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)

        return x


class DuelingNNEstimator(nn.Module):

    def __init__(self, action_num: int, input_ch: int = 4, nf: int = 32) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_ch, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 36, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 36, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.value_stream = nn.Sequential(
            nn.Linear(36 * 58 * 58, nf),
            nn.ReLU(inplace=True),
            nn.Linear(nf, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(36 * 58 * 58, nf),
            nn.ReLU(inplace=True),
            nn.Linear(nf, action_num)
        )

    def forward(self, x):
        x =  self.features(x)
        x = x.view(x.size(0), -1)

        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        q_values = values + (advantages - advantages.mean())
        return q_values