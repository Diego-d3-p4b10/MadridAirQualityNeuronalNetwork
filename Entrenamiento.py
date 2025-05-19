import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from tqdm import tqdm

# ------------------ CONFIGURACIÓN ------------------
RUTA = 'RUTA_AL_DF.csv'  # Cambia esto a la ruta de tu archivo CSV
VENTANA = 72
BATCH_SIZE = 128
EPOCHS = 30
TARGET = "VALOR_HORA"

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("No se detectó GPU, se usará CPU.")

# ------------------ CARGA Y PREPROCESADO ------------------
df = pd.read_csv(RUTA)
df = df.sort_values(by=["AÑO_cos", "MES_cos", "DIA_cos", "HORA_sin"])
df = df.select_dtypes(include=["number", "bool"]).astype(np.float32)
target_idx = df.columns.get_loc(TARGET)
data_np = df.to_numpy()
n = len(data_np)

# ------------------ DATASET ITERABLE ------------------
class VentanasDataset(IterableDataset):
    def __init__(self, data_array, target_idx, ventana):
        self.data = data_array
        self.target_idx = target_idx
        self.ventana = ventana

    def __iter__(self):
        for i in range(len(self.data) - self.ventana):
            x = self.data[i:i+self.ventana]
            y = self.data[i+self.ventana, self.target_idx]
            yield torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)

# ------------------ LOADERS ------------------
train_dataset = VentanasDataset(data_np[:int(n*0.7)], target_idx, VENTANA)
val_dataset   = VentanasDataset(data_np[int(n*0.7):int(n*0.8)], target_idx, VENTANA)
test_dataset  = VentanasDataset(data_np[int(n*0.8):], target_idx, VENTANA)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ------------------ MODELO ------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=32, dense_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(self.fc1(out[:, -1, :]))  # solo último paso
        return self.fc2(out)

model = LSTMRegressor(input_size=df.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------ ENTRENAMIENTO ------------------
def entrenar(model, loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        print(f"\nÉpoca {epoch+1}/{epochs}")
        progreso = tqdm(loader, desc="Entrenando", unit="batch")

        for x_batch, y_batch in progreso:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progreso.set_postfix(loss=loss.item())

        val_loss = evaluar(model, val_loader, criterion)
        print(f"Época {epoch+1} finalizada - Loss: {total_loss:.6f} - Val Loss: {val_loss:.6f}")

def evaluar(model, loader, criterion):
    model.eval()
    total_loss = 0
    total_batches = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            total_loss += criterion(pred, y_batch).item()
            total_batches += 1
    return total_loss / total_batches

# ------------------ EJECUTAR ENTRENAMIENTO ------------------
entrenar(model, train_loader, val_loader, optimizer, criterion, EPOCHS)

# ------------------ EVALUACIÓN FINAL ------------------
model.eval()
preds, trues = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        pred = model(x_batch).cpu().numpy()
        preds.extend(pred.flatten())
        trues.extend(y_batch.numpy().flatten())

mae = mean_absolute_error(trues, preds)
rmse = np.sqrt(mean_squared_error(trues, preds))
r2 = r2_score(trues, preds)

print("\nResultados finales del modelo:")
print(f"MAE :  {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"R²   :  {r2:.4f}")

# ------------------ GUARDADO DEL MODELO ------------------
os.makedirs("modelos", exist_ok=True)
torch.save(model.state_dict(), "modelos/modelo_lstm_valor_hora.pt")
print("Modelo guardado en 'modelos/modelo_lstm_valor_hora.pt'")
