from uuid import UUID
from pydantic import BaseModel
from typing import List, Literal, Optional, Union

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Craftsman(BaseModel):
    id: str  # CHAR(32)
    email: str
    password: str  # Note: Em produção, nunca retorne a senha!
    role: str
    name: str
    phone: str
    cpf: str
    active: bool
    photo_id: Optional[str] = None
    address_street: Optional[str] = None
    address_number: Optional[str] = None
    address_neighborhood: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None
    address_zip_code: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()  # Converte datetime para string no JSON
        }

class Product(BaseModel):
    id: str  # CHAR(32)
    name: str
    description: Optional[str] = None
    price: float
    type: str
    photo_url: Optional[str] = None
    # craftsman: Optional[Craftsman] = None  # Relacionamento opcional

    class Config:
        from_attributes = True

class ProductRecommendationRequest(BaseModel):
    product_id: str
    n_recommendations: Optional[int] = 5

class UserRecommendationRequest(BaseModel):
    customer_id: int
    n_recommendations: Optional[int] = 5

class RecommendationResponse(BaseModel):
    recommended_products: List[Product]
    
    
class ReactProductResponse(BaseModel):
    """
    Este schema representa um produto no formato exato
    que a interface 'Produto' do React espera.
    """
    id: str
    nome: str
    descricao: Optional[str] = None
    preco: float
    estoque: int
    imagemUrl: Optional[str] = None 
    status: Literal['ativo', 'inativo']
    categoria: str
    totalVendas: int

    class Config:
        from_attributes = True

class ReactRecommendationResponse(BaseModel):
    """
    Este é o objeto de resposta final que contém a lista de produtos
    no formato do React.
    """
    recommended_products: List[ReactProductResponse]