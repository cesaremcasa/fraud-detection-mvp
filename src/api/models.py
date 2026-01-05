"""
Schemas Pydantic para validação de dados da API.
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional
import uuid

class TransactionRequest(BaseModel):
    """Schema para requisição de transação de táxi."""
    
    # ID único da transação (gerado se não fornecido)
    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="ID único da transação (UUID)"
    )
    
    # Datetimes
    pickup_datetime: datetime = Field(
        ..., 
        description="Data e hora de embarque do passageiro"
    )
    
    dropoff_datetime: datetime = Field(
        ..., 
        description="Data e hora de desembarque do passageiro"
    )
    
    # Dados da corrida
    passenger_count: int = Field(
        ..., 
        ge=1, le=9,  # Mínimo 1, máximo 9 passageiros
        description="Número de passageiros (1-9)"
    )
    
    trip_distance: float = Field(
        ..., 
        gt=0, le=1000,  # Maior que 0, máximo 1000 milhas
        description="Distância da corrida em milhas (>0)"
    )
    
    fare_amount: float = Field(
        ..., 
        ge=0, le=1000,  # Mínimo 0, máximo $1000
        description="Valor da tarifa em dólares (≥0)"
    )
    
    # Campos opcionais para futuro
    payment_type: Optional[int] = Field(
        None,
        ge=1, le=6,
        description="Tipo de pagamento (1-6)"
    )
    
    vendor_id: Optional[int] = Field(
        None,
        ge=1, le=2,
        description="ID do provedor (1-2)"
    )
    
    @validator('dropoff_datetime')
    def validate_dropoff_after_pickup(cls, v, values):
        """Valida que dropoff é após pickup."""
        if 'pickup_datetime' in values and v <= values['pickup_datetime']:
            raise ValueError('dropoff_datetime must be after pickup_datetime')
        return v
    
    @validator('fare_amount')
    def validate_fare_reasonable(cls, v, values):
        """Validação básica de valor da tarifa."""
        if v == 0 and 'trip_distance' in values and values['trip_distance'] > 0.1:
            raise ValueError('fare_amount cannot be 0 for trips longer than 0.1 miles')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "123e4567-e89b-12d3-a456-426614174000",
                "pickup_datetime": "2023-01-01T00:26:10",
                "dropoff_datetime": "2023-01-01T00:37:11",
                "passenger_count": 1,
                "trip_distance": 2.58,
                "fare_amount": 24.18,
                "payment_type": 1,
                "vendor_id": 2
            }
        }

class TransactionResponse(BaseModel):
    """Schema para resposta da API."""
    
    status: str = Field(..., description="Status da operação")
    transaction_id: str = Field(..., description="ID da transação")
    message: Optional[str] = Field(None, description="Mensagem adicional")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "accepted",
                "transaction_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "Transaction queued for processing",
                "timestamp": "2023-01-01T00:37:11.123456"
            }
        }

class HealthResponse(BaseModel):
    """Schema para resposta de health check."""
    
    status: str = Field(..., description="Status do serviço")
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = Field("1.0.0", description="Versão da API")
    kafka_connected: bool = Field(..., description="Status da conexão Kafka")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2023-01-01T00:37:11.123456",
                "version": "1.0.0",
                "kafka_connected": True
            }
        }
