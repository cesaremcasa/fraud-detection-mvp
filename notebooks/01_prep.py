#!/usr/bin/env python3
"""
Script de preparação de dados para detecção de fraude em táxis NYC.
Bloco 1: Engenharia de Features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
from pathlib import Path

def main():
    print("=== BLOCO 1: Preparação de Dados e Engenharia de Features ===\n")
    
    # Configurar paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    artifacts_dir = project_root / "artifacts"
    
    # Criar diretórios se não existirem
    artifacts_dir.mkdir(exist_ok=True)
    
    # 1. Caminhos dos arquivos
    raw_data_path = data_dir / "green_tripdata_2023-01.parquet"
    processed_sample_path = data_dir / "processed_sample.parquet"
    scaler_path = artifacts_dir / "scaler.pkl"
    
    print(f"1. Carregando dados de: {raw_data_path}")
    
    # 2. Carregar dados - APENAS COLUNAS NECESSÁRIAS
    try:
        # Carregar apenas colunas necessárias para evitar problemas com colunas vazias
        cols_to_load = [
            'lpep_pickup_datetime', 
            'lpep_dropoff_datetime',
            'passenger_count', 
            'trip_distance', 
            'fare_amount'
        ]
        
        df = pd.read_parquet(raw_data_path, columns=cols_to_load)
        print(f"   ✅ Dataset carregado. Shape original: {df.shape}")
        print(f"   Colunas carregadas: {list(df.columns)}")
    except Exception as e:
        print(f"   ❌ Erro ao carregar dados: {e}")
        sys.exit(1)
    
    # 3. Limpeza de Dados - ABORDAGEM MAIS CONSERVADORA
    print("\n2. Executando limpeza de dados...")
    
    # Fazer cópia para não modificar o original
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    # Remover linhas onde as colunas essenciais são nulas
    essential_cols = ['passenger_count', 'trip_distance', 'fare_amount']
    df_clean = df_clean.dropna(subset=essential_cols)
    removed_nulls = initial_rows - len(df_clean)
    print(f"   • Removidas {removed_nulls} linhas com valores nulos nas colunas essenciais")
    
    # Remover linhas com passenger_count = 0
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['passenger_count'] > 0]
    removed_passenger = initial_rows - len(df_clean)
    print(f"   • Removidas {removed_passenger} linhas com passenger_count <= 0")
    
    # Remover linhas com trip_distance = 0
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['trip_distance'] > 0]
    removed_distance = initial_rows - len(df_clean)
    print(f"   • Removidas {removed_distance} linhas com trip_distance <= 0")
    
    # Remover linhas com fare_amount negativo
    initial_rows = len(df_clean)
    df_clean = df_clean[df_clean['fare_amount'] >= 0]
    removed_fare = initial_rows - len(df_clean)
    print(f"   • Removidas {removed_fare} linhas com fare_amount negativo")
    
    print(f"   ✅ Dataset após limpeza: {df_clean.shape}")
    print(f"   Linhas restantes: {len(df_clean)} / {len(df)} ({len(df_clean)/len(df)*100:.1f}%)")
    
    # 4. Engenharia de Features
    print("\n3. Engenharia de features...")
    
    # Converter colunas de tempo
    df_clean['lpep_pickup_datetime'] = pd.to_datetime(df_clean['lpep_pickup_datetime'])
    df_clean['lpep_dropoff_datetime'] = pd.to_datetime(df_clean['lpep_dropoff_datetime'])
    
    # Calcular trip_duration_min
    df_clean['trip_duration_min'] = (
        df_clean['lpep_dropoff_datetime'] - df_clean['lpep_pickup_datetime']
    ).dt.total_seconds() / 60.0
    
    # Remover durações não positivas ou muito longas (supostos erros)
    initial_rows = len(df_clean)
    df_clean = df_clean[
        (df_clean['trip_duration_min'] > 0) & 
        (df_clean['trip_duration_min'] <= 180)  # max 3 horas
    ]
    removed_duration = initial_rows - len(df_clean)
    print(f"   • Removidas {removed_duration} linhas com trip_duration inválida")
    
    # Calcular fare_per_minute
    df_clean['fare_per_minute'] = df_clean['fare_amount'] / df_clean['trip_duration_min']
    
    # Remover fare_per_minute extremos (supostos outliers/erros)
    initial_rows = len(df_clean)
    df_clean = df_clean[
        (df_clean['fare_per_minute'] > 0) & 
        (df_clean['fare_per_minute'] <= 50)  # max $50/min
    ]
    removed_fare_rate = initial_rows - len(df_clean)
    print(f"   • Removidas {removed_fare_rate} linhas com fare_per_minute extremo")
    
    print(f"   ✅ Dataset após feature engineering: {df_clean.shape}")
    print(f"   Linhas restantes: {len(df_clean)} / {len(df)} ({len(df_clean)/len(df)*100:.1f}%)")
    
    # 5. Seleção de Features
    print("\n4. Seleção de features...")
    
    feature_columns = [
        'passenger_count', 
        'trip_distance', 
        'fare_amount', 
        'trip_duration_min', 
        'fare_per_minute'
    ]
    
    # Criar dataframe final com features
    df_features = df_clean[feature_columns].copy()
    print(f"   ✅ Features selecionadas: {feature_columns}")
    print(f"   Shape final: {df_features.shape}")
    
    if len(df_features) == 0:
        print("   ❌ ERRO: Nenhum dado restante após limpeza!")
        print("   Verificando estatísticas...")
        print(df_clean.describe())
        sys.exit(1)
    
    # 6. Scaling e Salvamento
    print("\n5. Scaling e salvamento de artefatos...")
    
    # Instanciar e treinar scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # Criar dataframe escalado
    df_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=df_features.index)
    
    # Salvar scaler
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Scaler salvo em: {scaler_path}")
    
    # Salvar amostra processada
    sample_size = min(100, len(df_scaled))
    df_scaled_sample = df_scaled.iloc[:sample_size].copy()
    df_scaled_sample.to_parquet(processed_sample_path)
    print(f"   ✅ Amostra salva em: {processed_sample_path} ({sample_size} linhas)")
    
    # 7. Verificação e Validação
    print("\n" + "="*60)
    print("VERIFICAÇÃO FINAL:")
    print("="*60)
    
    print(f"\n• Shape do dataset original: {df.shape}")
    print(f"• Shape do dataset após limpeza: {df_features.shape}")
    print(f"• Total de linhas removidas: {len(df) - len(df_features)}")
    print(f"• Percentual mantido: {len(df_features)/len(df)*100:.1f}%")
    
    print("\n• Estatísticas ANTES do scaling:")
    print(df_features.describe().round(4))
    
    print("\n• Estatísticas APÓS scaling (deveria ser ~0 média, ~1 desvio):")
    stats = df_scaled.describe().loc[['mean', 'std']]
    print(stats.round(4))
    
    print("\n• Verificação dos valores salvos:")
    print(f"  Scaler: {scaler_path} - Existe: {scaler_path.exists()}")
    print(f"  Amostra: {processed_sample_path} - Existe: {processed_sample_path.exists()}")
    
    print("\n" + "="*60)
    print("✅ BLOCO 1 CONCLUÍDO COM SUCESSO!")
    print("="*60)

if __name__ == "__main__":
    main()
