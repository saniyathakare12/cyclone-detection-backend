import torch
import torch.nn as nn
import numpy as np


class CycloneLSTMMultiStep(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, horizon=6, output_features=4, dropout=0.2):
        super().__init__()
        self.horizon = horizon
        self.output_features = output_features
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon * output_features)

    def forward(self, x):
        out, (hidden, cell) = self.lstm(x)
        last_output = out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.view(-1, self.horizon, self.output_features)


def load_cyclone_model(checkpoint_path, device='cpu'):
    model = CycloneLSTMMultiStep(horizon=6)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def predict_cyclone_path(model, checkpoint, recent_readings, device='cpu'):
    if len(recent_readings) != 3:
        raise ValueError(f"Expected exactly 3 readings, got {len(recent_readings)}")

    feature_mean = checkpoint['feature_mean']
    feature_std = checkpoint['feature_std']
    delta_mean = checkpoint['delta_mean']
    delta_std = checkpoint['delta_std']

    seq = np.array(recent_readings)
    seq_norm = (seq - feature_mean) / feature_std
    seq_tensor = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_output = model(seq_tensor).cpu().numpy()[0]

    pred_deltas_real = pred_output * delta_std + delta_mean
    last_observed = seq[-1]

    predictions = []
    for h in range(6):
        pred = last_observed + pred_deltas_real[h]
        predictions.append({
            "hours_ahead": (h + 1) * 3,
            "lat": float(pred[0]),
            "lon": float(pred[1]),
            "wind": float(pred[2]),
            "pressure": float(pred[3]),
        })
    return predictions
