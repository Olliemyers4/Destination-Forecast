import types

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray) -> None:
        self.data = torch.from_numpy(data.astype("float32"))
        self.label = torch.from_numpy(label).long()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.label[index]


class LSTMForecaster(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        lstm_layers: int,
        output_size: int,
        activation: types.FunctionType,
    ) -> None:
        super(LSTMForecaster, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, lstm_layers)
        self.fc1 = torch.nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.lstm(x))  # type: ignore
        x = self.fc1(x)
        return x


class ForecasterTrainer:
    def __init__(
        self,
        model: LSTMForecaster,
        batch_size: int,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        valid_x: np.ndarray,
        valid_y: np.ndarray,
        optimiser: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: str,
        mlflow: types.ModuleType,
    ) -> None:
        self.model = model.to(device)
        self.batch_size = batch_size

        self.train_x = train_x
        self.train_y = train_y
        self.train = Dataset(train_x, train_y)
        self.train_loader = torch.utils.data.DataLoader(
            self.train, batch_size=self.batch_size
        )

        self.test_x = test_x
        self.test_y = test_y
        self.test = Dataset(test_x, test_y)
        self.test_loader = torch.utils.data.DataLoader(
            self.test, batch_size=self.batch_size
        )

        self.valid_x = valid_x
        self.valid_y = valid_y
        self.valid = Dataset(valid_x, valid_y)
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid, batch_size=self.batch_size
        )

        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device
        self.mlflow = mlflow

    def train_model(self, epochs: int) -> None:
        for _ in range(epochs):
            loss = 0.0
            for _, (data, label) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                self.optimiser.zero_grad()
                output = self.model(data)
                losses = self.criterion(output, label)
                losses.backward()
                self.optimiser.step()
                loss += losses.item()
            self.validate_model()

    def validate_model(self) -> float:
        self.model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for _, (data, label) in enumerate(self.valid_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.model(data)
                losses = self.criterion(output, label)
                validation_loss += losses.item()
        self.model.train()
        return validation_loss
