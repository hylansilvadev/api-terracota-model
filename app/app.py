import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# --- ETAPA 1: CONFIGURAÇÃO DE LOGS (DEVE VIR PRIMEIRO) ---
# Isto garante que qualquer log gerado durante a inicialização seja capturado.
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# Handler para salvar logs em um arquivo
file_handler = RotatingFileHandler('recommendation_api.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
# Handler para mostrar logs no terminal
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
# Pega o logger principal e adiciona os handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


# --- ETAPA 2: IMPORTS DO PROJETO E INICIALIZAÇÃO DO RECOMENDADOR ---
# Agora que os logs estão configurados, podemos importar nossos módulos com segurança.
from .models import (
    Product, ProductRecommendationRequest, ReactProductResponse, 
    ReactRecommendationResponse, UserRecommendationRequest
)
from .recommender import AdvancedProductRecommender
from .database import database
from .queries import QUERIES

# O recomendador é inicializado aqui, e os logs do seu treino serão capturados.
product_recommender = AdvancedProductRecommender()


# --- ETAPA 3: CONFIGURAÇÃO DA API FASTAPI ---
app = FastAPI(
    title="Sistema de Recomendação de Produtos Artesanais",
    description="API para recomendar produtos artesanais usando KNN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"],   
    allow_headers=["*"],    
)

# ... (O restante do seu código de endpoints continua aqui, sem alterações) ...
@app.post("/recommend/user/", response_model=ReactRecommendationResponse)
async def recommend_for_user_endpoint(request: UserRecommendationRequest):
    """
    Recomenda produtos para um cliente com base no seu histórico de compras
    e no comportamento de usuários similares.
    """
    try:
        recommended_ids = product_recommender.recommend_for_user(
            request.customer_id,
            request.n_recommendations
        )
        if not recommended_ids:
            return ReactRecommendationResponse(recommended_products=[])
        with database.get_cursor() as cursor:
            # A query get_similar_products usa 'id', que é a chave primária de 'products'
            # A CTE dentro dela garante que pegamos todos os detalhes do produto.
            # Essa parte está correta.
            cursor.execute(QUERIES["get_similar_products"], (list(recommended_ids),))
            recommended_products = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        products_list = [
            ReactProductResponse(
                id=str(product[columns.index('id')]),
                nome=product[columns.index('name')],
                descricao=product[columns.index('description')],
                preco=float(product[columns.index('price')]),
                estoque=int(product[columns.index('quantity')]),
                imagemUrl=product[columns.index('photo_url')],
                status='ativo',
                categoria=product[columns.index('type')],
                totalVendas=0
            ) for product in recommended_products
        ]
        return ReactRecommendationResponse(recommended_products=products_list)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao gerar recomendações para o usuário: {str(e)}")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar recomendações")

# (Cole aqui seus outros endpoints: /products e /recommend/products)


@app.on_event("startup")
async def startup_event():
    logger.info("Serviço de recomendação iniciado e pronto para receber requisições.")
    # Forçar avaliação de modelo no início
    try:
        product_recommender.evaluate_model_precision()
    except Exception as e:
        logger.error(f"Erro ao avaliar modelo durante o startup: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Este modo de execução é útil para alguns tipos de deploy, mas para desenvolvimento,
    # o comando 'uvicorn app.app:app --reload' é o preferido.
    uvicorn.run(app, host="0.0.0.0", port=8000)