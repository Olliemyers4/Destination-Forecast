import mlflow
import torch

from dataset_preperation.dataset_data_label_split import data_split
from forecaster import ForecasterTrainer, LSTMForecaster

test_x, test_y = data_split("test.csv", 3)

train_x, train_y = data_split("train.csv", 3)

valid_x, valid_y = data_split("validation.csv", 3)

mlflow.set_experiment("Port Forecasting")
mlflow.config.enable_system_metrics_logging()
mlflow.config.set_system_metrics_sampling_interval(1)

# Config
input_size = 7
hidden_size = 5
lstm_layers = 2
output_size = 3

with mlflow.start_run():

    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        output_size=output_size,
        activation=torch.nn.ReLU(),
    )  # type: ignore
    mlflow.log_params(
        {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "lstm_layers": lstm_layers,
            "output_size": output_size,
            "activation": "ReLU",
        }
    )
    trainer = ForecasterTrainer(
        model=model,
        batch_size=100,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        valid_x=valid_x,
        valid_y=valid_y,
        optimiser=torch.optim.Adam(model.parameters(), lr=0.005),
        criterion=torch.nn.MSELoss(),
        device="cpu",
        mlflow=mlflow,
    )

    trainer.train_model(30, "lstm_forecaster")
