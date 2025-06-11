from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
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

origins = ["*"]


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
    Recomenda produtos similares com base em um ID de produto.
    """
    try:
        # 1. Buscar as features do produto de referência pelo ID
        with database.get_cursor() as cursor:
            cursor.execute(QUERIES["get_product_features"] + " WHERE id = %s", (request.product_id,))
            product_data = cursor.fetchone()
            if not product_data:
                raise ValueError(f"Produto com ID {request.product_id} não encontrado")
            columns = [desc[0] for desc in cursor.description]
        
        # Converter para DataFrame, o formato que o recomendador espera
        product_features_df = pd.DataFrame([product_data], columns=columns)

        # 2. Obter as recomendações
        recommended_ids = product_recommender.recommend_similar_products(
            product_features_df,
            request.n_recommendations
        )
        
        if not recommended_ids:
            return ReactRecommendationResponse(recommended_products=[])
        
        # 3. Buscar os dados completos dos produtos recomendados para a resposta
        # (Seu código para buscar os produtos pelo ID e formatar a resposta já está ótimo)
        with database.get_cursor() as cursor:
            final_query = QUERIES["get_similar_products"].format(
                product_details_cte=QUERIES["product_details_cte"]
            )
            # Psycopg2 espera uma tupla para o 'IN'
            cursor.execute(final_query, (tuple(recommended_ids),))
            
            recommended_products = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

        products_list = []
        # (Seu código de formatação da resposta continua aqui...)
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
    """Atividades de inicialização: Carrega dados, avalia e treina o modelo."""
    logger.info("🚀 Serviço de recomendação iniciando...")
    
    # ... (verificação de conexão com o banco)

    logger.info("🔄 Carregando dados dos produtos para o modelo...")
    try:
        with database.get_cursor() as cursor:
            cursor.execute(QUERIES["get_product_features"])
            products_data = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        if not products_data:
            logger.warning("⚠️ Nenhum produto encontrado no banco. O modelo não será treinado.")
            return

        all_products_df = pd.DataFrame(products_data, columns=columns)
        
        # --- ENGENHARIA DE FEATURE APLICADA AQUI ---
        # 1. Criar a coluna de texto combinado
        logger.info("🛠️  Aplicando engenharia de features: combinando nome e descrição.")
        all_products_df['text_feature'] = all_products_df['name'] + ' ' + all_products_df['description']

        logger.info(f"📊 {len(all_products_df)} produtos carregados. Amostra dos dados:")
        # Mostra a nova coluna na "visualização"
        print(all_products_df[['id', 'type', 'text_feature']].head().to_string())

        # 2. Avaliar o modelo com o DataFrame modificado
        product_recommender.evaluate(all_products_df, test_size=0.2, k=5)

        # 3. Treinar o modelo final com TODOS os dados
        logger.info("🎓 Treinando o modelo final com todos os dados disponíveis...")
        product_recommender._train_model(all_products_df)
        logger.info("✅ Modelo de recomendação pronto para receber requisições!")

    except Exception as e:
        logger.error(f"❌ Falha crítica durante a inicialização: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)