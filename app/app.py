from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import (
    Product,
    ProductRecommendationRequest,
    ReactProductResponse,
    ReactRecommendationResponse,
    UserRecommendationRequest,
    RecommendationResponse
)
from .recommender import ProductRecommender
from .database import database
from .queries import QUERIES
import logging
from typing import List

app = FastAPI(
    title="Sistema de Recomendação de Produtos Artesanais",
    description="API para recomendar produtos artesanais usando KNN",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "https://www.terracota.vercel.app",
    "https://spring-terracota-new.onrender.com",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True, 
    allow_methods=["*"],   
    allow_headers=["*"],    
)

product_recommender = ProductRecommender()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Função auxiliar para garantir o formato do ID
def format_char32_id(raw_id):
    if isinstance(raw_id, str):
        return raw_id.strip()  # Remove espaços em branco
    elif isinstance(raw_id, bytes):
        return raw_id.decode('utf-8').strip()
    return str(raw_id)


def fetch_products(product_ids: list) -> List[Product]:
    if not product_ids:
        return []
    
    with database.get_cursor() as cursor:
        # Converta a lista de IDs em uma tupla para a consulta SQL
        cursor.execute(QUERIES["get_similar_products"], (tuple(product_ids),))
        products = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    
    return [
        Product(
            id=product[columns.index('id')],
            name=product[columns.index('name')],
            description=product[columns.index('description')],
            price=product[columns.index('price')],
            type=product[columns.index('type')],
            photo_url=product[columns.index('photo_url')],
            craftsman_name=product[columns.index('craftsman_name')]
        ) for product in products
    ]

@app.get("/products/", response_model=List[Product])
async def get_products():
    """Retorna todos os produtos disponíveis com informações básicas do artesão"""
    try:
        with database.get_cursor() as cursor:
            cursor.execute(QUERIES["get_all_products"])
            products = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        product_list = []
        for product in products:
            product_data = {
                "id": str(product[columns.index('id')]),
                "name": product[columns.index('name')],
                "description": product[columns.index('description')],
                "price": float(product[columns.index('price')]),
                "type": product[columns.index('type')],
                "photo_url": product[columns.index('photo_url')]
            }
            
            # # Adiciona informações básicas do artesão se existirem
            # if 'craftsman_id' in columns and product[columns.index('craftsman_id')]:
            #     product_data["craftsman"] = {
            #         "id": str(product[columns.index('craftsman_id')]),
            #         "name": product[columns.index('craftsman_name')],
            #         "email": product[columns.index('craftsman_email')]
            #     }
            product_data["id"] = format_char32_id(product[columns.index('id')])
            product_list.append(Product(**product_data))
        
        return product_list
    except Exception as e:
        logger.error(f"Erro ao buscar produtos: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao buscar produtos")

@app.post("/recommend/products/", response_model=ReactRecommendationResponse)
async def recommend_products(request: ProductRecommendationRequest):
    """
    Recomenda produtos similares, retornando no formato do React.
    """
    try:
        recommended_ids = product_recommender.recommend_similar_products(
            request.product_id,
            request.n_recommendations
        )
        
        if not recommended_ids:
            return ReactRecommendationResponse(recommended_products=[])
        
        with database.get_cursor() as cursor:
            # ✅ CORREÇÃO APLICADA AQUI
            # 1. Monta a string da query final
            final_query = QUERIES["get_similar_products"].format(
                product_details_cte=QUERIES["product_details_cte"]
            )
            
            # 2. Executa a query final
            cursor.execute(final_query, (recommended_ids,))
            
            recommended_products = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        products_list = []
        for product in recommended_products:
            
            product_data = ReactProductResponse(
                id=str(product[columns.index('id')]),
                nome=product[columns.index('name')],
                descricao=product[columns.index('description')],
                preco=float(product[columns.index('price')]),
                estoque=int(product[columns.index('quantity')]), 
                imagemUrl=product[columns.index('photo_url')],  
                status='ativo',  
                categoria=product[columns.index('type')],      
                totalVendas=0 
            )
            products_list.append(product_data)
        
        return ReactRecommendationResponse(recommended_products=products_list)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao gerar recomendações: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar recomendações")

@app.on_event("startup")
async def startup_event():
    """Atividades de inicialização"""
    logger.info("Iniciando serviço de recomendação...")
    # Verificar conexão com o banco de dados
    try:
        with database.get_connection():
            logger.info("Conexão com o banco de dados estabelecida com sucesso")
    except Exception as e:
        logger.error(f"Erro ao conectar ao banco de dados: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)