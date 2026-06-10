import types

import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: np.ndarray, label: np.ndarray) -> None:
        self.data = torch.from_numpy(data.astype("float32"))
        self.label = torch.from_numpy(label).float()

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
        activation: torch.nn.Module,
    ) -> None:
        super(LSTMForecaster, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, lstm_layers, batch_first=True
        )
        self.fc1 = torch.nn.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)  # type: ignore
        x = self.activation(self.fc1(x[:, -1, :]))
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

    def train_model(self, epochs: int, name: str) -> None:
        for epoch in range(epochs):
            self.model.train()
            loss = 0.0
            batches = 0
            for _, (data, label) in enumerate(self.train_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                self.optimiser.zero_grad()
                output = self.model(data)
                losses = self.criterion(output, label)
                losses.backward()
                self.optimiser.step()
                loss += losses.item()
                batches += 1
            validation_loss, validation_batches = self.validate_model()
            self.mlflow.log_metrics(
                {
                    "training_loss": loss / batches,
                    "validation_loss": validation_loss / validation_batches,
                },
                step=epoch,
            )
        torch.save(self.model.state_dict(), f"{name}.pth")
        correct = self.evaluate_model()
        self.mlflow.log_metrics(
            {"final_score": correct / self.test_loader.__len__()}, step=epochs
        )

    def validate_model(self) -> tuple[float, int]:
        self.model.eval()
        validation_loss = 0.0
        validation_batches = 0
        with torch.no_grad():
            for _, (data, label) in enumerate(self.valid_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                output = self.model(data)
                losses = self.criterion(output, label)
                validation_loss += losses.item()
                validation_batches += 1
        self.model.train()
        return validation_loss, validation_batches

    def evaluate_model(self) -> int:
        self.model.eval()
        with torch.no_grad():
            correct = 0
            for i, (data, label) in enumerate(self.test_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                predicted = self.model(data)
                tol_lat = 0.01  # very rough approximates
                tol_lon = 0.01

                diff = (predicted[:, :2] - label[:, :2]).abs()

                correct = ((diff[:, 0] < tol_lat) & (diff[:, 1] < tol_lon)).sum().item()
        return correct
