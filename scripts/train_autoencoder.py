#!/usr/bin/env python3
"""
Bloco 2: Treinamento do Autoencoder para Detecção de Anomalias
VERSÃO CORRIGIDA - Sem matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BLOCO 2: Treinamento do Modelo de Detecção de Anomalias")
print("="*70)

# ============================================================================
# 1. CONFIGURAÇÃO
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n1. Configurando dispositivo: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memória GPU disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# 2. CARREGAMENTO E PREPARAÇÃO DE DADOS
# ============================================================================
print("\n2. Carregando e preparando dados...")

# Caminhos
project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "green_tripdata_2023-01.parquet"
scaler_path = project_root / "artifacts" / "scaler.pkl"
artifacts_dir = project_root / "artifacts"

# Carregar scaler
scaler = joblib.load(scaler_path)
print(f"   ✅ Scaler carregado de: {scaler_path}")
print(f"   • Número de features: {scaler.n_features_in_}")

# Carregar dados originais e reprocessar (para consistência)
df_raw = pd.read_parquet(data_path)

# Aplicar pipeline de limpeza do Bloco 1
# 1. Remover valores nulos nas colunas essenciais
cols = ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'fare_amount']
df_clean = df_raw[cols].dropna()

# 2. Filtrar valores inválidos
df_clean = df_clean[
    (df_clean['passenger_count'] > 0) &
    (df_clean['trip_distance'] > 0) &
    (df_clean['fare_amount'] >= 0)
]

# 3. Engenharia de features
df_clean['lpep_pickup_datetime'] = pd.to_datetime(df_clean['lpep_pickup_datetime'])
df_clean['lpep_dropoff_datetime'] = pd.to_datetime(df_clean['lpep_dropoff_datetime'])
df_clean['trip_duration_min'] = (df_clean['lpep_dropoff_datetime'] - df_clean['lpep_pickup_datetime']).dt.total_seconds() / 60.0
df_clean = df_clean[
    (df_clean['trip_duration_min'] > 0) & 
    (df_clean['trip_duration_min'] <= 180)
]
df_clean['fare_per_minute'] = df_clean['fare_amount'] / df_clean['trip_duration_min']
df_clean = df_clean[
    (df_clean['fare_per_minute'] > 0) & 
    (df_clean['fare_per_minute'] <= 50)
]

# 4. Selecionar features
features = ['passenger_count', 'trip_distance', 'fare_amount', 'trip_duration_min', 'fare_per_minute']
df_features = df_clean[features].copy()

print(f"   • Dataset original: {len(df_raw)} linhas")
print(f"   • Dataset após limpeza: {len(df_features)} linhas ({len(df_features)/len(df_raw)*100:.1f}%)")

# ============================================================================
# 3. FILTRAGEM DE "NORMALIDADE" (remover outliers extremos)
# ============================================================================
print("\n3. Filtrando dados para treino (percentis 5-95 de fare_per_minute)...")

# Calcular percentis
low_percentile = df_features['fare_per_minute'].quantile(0.05)
high_percentile = df_features['fare_per_minute'].quantile(0.95)

# Filtrar dados "normais"
mask_normal = (df_features['fare_per_minute'] >= low_percentile) & (df_features['fare_per_minute'] <= high_percentile)
df_normal = df_features[mask_normal].copy()

print(f"   • Percentil 5: {low_percentile:.4f}")
print(f"   • Percentil 95: {high_percentile:.4f}")
print(f"   • Dados normais: {len(df_normal)} linhas ({len(df_normal)/len(df_features)*100:.1f}%)")

# Aplicar scaling
X_normal = scaler.transform(df_normal)
print(f"   • Dados escalados: {X_normal.shape}")

# ============================================================================
# 4. SEPARAÇÃO TREINO/VALIDAÇÃO
# ============================================================================
print("\n4. Separando dados em treino (80%) e validação (20%)...")

X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

# Converter para tensores PyTorch
X_train_tensor = torch.FloatTensor(X_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)

# Criar DataLoaders
train_dataset = TensorDataset(X_train_tensor)
val_dataset = TensorDataset(X_val_tensor)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"   • Treino: {len(X_train)} amostras")
print(f"   • Validação: {len(X_val)} amostras")
print(f"   • Batch size: {batch_size}")

# ============================================================================
# 5. DEFINIÇÃO DA ARQUITETURA DO AUTOENCODER
# ============================================================================
print("\n5. Definindo arquitetura do Autoencoder...")

class Autoencoder(nn.Module):
    def __init__(self, input_dim=5, latent_dim=2):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Instanciar modelo
input_dim = X_train.shape[1]
model = Autoencoder(input_dim=input_dim, latent_dim=2).to(device)

# Otimizador e função de perda
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(f"   • Dimensão de entrada: {input_dim}")
print(f"   • Dimensão latente: 2")
print(f"   • Otimizador: Adam (lr=0.001)")
print(f"   • Loss function: MSELoss")
print(f"   • Total de parâmetros: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 6. LOOP DE TREINAMENTO
# ============================================================================
print("\n6. Iniciando treinamento...")
print("-" * 70)

epochs = 30  # Reduzido para teste rápido
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Modo de treino
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        data = batch[0]
        
        # Forward pass
        reconstructed, _ = model(data)
        loss = criterion(reconstructed, data)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
    
    # Calcular média do loss de treino
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)
    
    # Modo de avaliação
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            data = batch[0]
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            val_loss += loss.item() * data.size(0)
    
    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    
    # Progresso a cada 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"   Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f}")

print("-" * 70)
print(f"   ✅ Treinamento concluído!")

# ============================================================================
# 7. CÁLCULO DO THRESHOLD DE ANOMALIA
# ============================================================================
print("\n7. Calculando threshold de anomalia...")

model.eval()
reconstruction_errors = []

with torch.no_grad():
    for batch in val_loader:
        data = batch[0]
        reconstructed, _ = model(data)
        
        # Calcular MSE para cada amostra
        mse = torch.mean((reconstructed - data) ** 2, dim=1)
        reconstruction_errors.extend(mse.cpu().numpy())

reconstruction_errors = np.array(reconstruction_errors)

# Definir threshold como percentil 95
threshold = np.percentile(reconstruction_errors, 95)

print(f"   • Erro médio de reconstrução: {np.mean(reconstruction_errors):.6f}")
print(f"   • Erro mínimo: {np.min(reconstruction_errors):.6f}")
print(f"   • Erro máximo: {np.max(reconstruction_errors):.6f}")
print(f"   • Percentil 95 (threshold): {threshold:.6f}")

# Salvar threshold
thresholds_data = {
    "autoencoder_mse_threshold": float(threshold),
    "autoencoder_mean_error": float(np.mean(reconstruction_errors)),
    "autoencoder_std_error": float(np.std(reconstruction_errors)),
    "percentile_used": 95,
    "validation_samples": len(reconstruction_errors)
}

thresholds_path = artifacts_dir / "thresholds.json"
with open(thresholds_path, 'w') as f:
    json.dump(thresholds_data, f, indent=2)

print(f"   ✅ Threshold salvo em: {thresholds_path}")

# ============================================================================
# 8. SALVAMENTO DO MODELO
# ============================================================================
print("\n8. Salvando modelo...")

# Salvar state_dict do modelo
model_path = artifacts_dir / "autoencoder.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'latent_dim': 2,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'threshold': threshold
}, model_path)

print(f"   ✅ Modelo salvo em: {model_path}")

# ============================================================================
# 9. MODELO ESTATÍSTICO (Isolation Forest - Opcional)
# ============================================================================
print("\n9. Treinando Isolation Forest (opcional)...")

try:
    # Usar dados de treino não escalados para Isolation Forest
    X_train_original = scaler.inverse_transform(X_train)
    
    # Treinar Isolation Forest
    iso_forest = IsolationForest(
        contamination=0.01,  # Espera-se 1% de anomalias
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    
    iso_forest.fit(X_train_original)
    
    # Salvar modelo
    iso_path = artifacts_dir / "iforest.pkl"
    joblib.dump(iso_forest, iso_path)
    
    # Fazer predição nos dados de validação
    X_val_original = scaler.inverse_transform(X_val)
    val_predictions = iso_forest.predict(X_val_original)
    anomaly_ratio = np.sum(val_predictions == -1) / len(val_predictions)
    
    print(f"   ✅ Isolation Forest treinado")
    print(f"   • Contamination esperada: 0.01")
    print(f"   • Anomalias detectadas na validação: {anomaly_ratio:.4f}")
    print(f"   • Modelo salvo em: {iso_path}")
    
except Exception as e:
    print(f"   ⚠️  Isolation Forest não pôde ser treinado: {e}")

# ============================================================================
# 10. VALIDAÇÃO E VISUALIZAÇÃO
# ============================================================================
print("\n10. Validação do modelo...")

# Testar reconstrução em algumas amostras normais
model.eval()
with torch.no_grad():
    test_samples = X_val_tensor[:5]
    reconstructed, latent = model(test_samples)
    
    # Calcular erro de reconstrução
    mse_per_sample = torch.mean((reconstructed - test_samples) ** 2, dim=1)
    
    print(f"   • Testando 5 amostras normais:")
    for i, (mse, sample, recon) in enumerate(zip(mse_per_sample, test_samples, reconstructed)):
        is_anomaly = mse > threshold
        status = "ANOMALIA" if is_anomaly else "normal"
        print(f"     Amostra {i+1}: MSE = {mse:.6f} ({status})")

# Verificar distribuição dos erros
print(f"\n   • Distribuição dos erros de reconstrução:")
print(f"     - Abaixo do threshold: {(reconstruction_errors <= threshold).sum()} amostras")
print(f"     - Acima do threshold (anomalias): {(reconstruction_errors > threshold).sum()} amostras")
print(f"     - Percentual de anomalias detectadas: {(reconstruction_errors > threshold).sum() / len(reconstruction_errors) * 100:.2f}%")

# ============================================================================
# 11. SALVAR HISTÓRICO DE TREINAMENTO
# ============================================================================
history_path = artifacts_dir / "training_history.json"
history_data = {
    "train_losses": [float(loss) for loss in train_losses],
    "val_losses": [float(loss) for loss in val_losses],
    "final_train_loss": float(train_losses[-1]),
    "final_val_loss": float(val_losses[-1]),
    "epochs": epochs,
    "batch_size": batch_size
}

with open(history_path, 'w') as f:
    json.dump(history_data, f, indent=2)

print(f"\n   ✅ Histórico de treinamento salvo em: {history_path}")

# ============================================================================
print("\n" + "="*70)
print("✅ BLOCO 2 CONCLUÍDO COM SUCESSO!")
print("="*70)
print("\nArtefatos gerados:")
print(f"  • {model_path.name} - Modelo Autoencoder treinado")
print(f"  • {thresholds_path.name} - Thresholds para detecção")
print(f"  • {history_path.name} - Histórico de treinamento")
if Path(artifacts_dir / "iforest.pkl").exists():
    print(f"  • iforest.pkl - Modelo Isolation Forest")
print(f"\nPronto para o Bloco 3: API FastAPI + Kafka Producer!")
