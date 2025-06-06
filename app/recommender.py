import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from .database import database
from .queries import QUERIES
import logging
from nltk.corpus import stopwords
import nltk

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download das stop words (opcional)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ProductRecommender:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.product_ids = None
        self.product_features = None
        self._train_model()
    
    def _train_model(self):
        logger.info("Treinando modelo de recomendação de produtos...")
        
        # Buscar dados do banco
        with database.get_cursor() as cursor:
            cursor.execute(QUERIES["get_product_features"])
            products = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
        
        if not products:
            logger.warning("Nenhum produto encontrado para treinar o modelo")
            return
        
        # Criar DataFrame
        products_df = pd.DataFrame(products, columns=columns)
        self.product_ids = products_df['id'].values
        self.product_features = products_df
        
        # Pré-processamento com stop words em português
        preprocessor = ColumnTransformer(
            transformers=[
                ('desc', TfidfVectorizer(
                    stop_words=stopwords.words('portuguese'),
                    max_features=1000,
                    min_df=2,
                    max_df=0.95
                ), 'description'),
                ('type', OneHotEncoder(handle_unknown='ignore'), ['type']),
                ('price', StandardScaler(), ['price'])
            ],
            remainder='drop'
        )
        
        # Pipeline de processamento
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
        ])
        
        # Transformar dados
        features = pipeline.fit_transform(products_df)
        
        # Treinar modelo KNN
        self.model = NearestNeighbors(
            n_neighbors=11,
            metric='cosine',
            algorithm='brute'
        )
        self.model.fit(features)
        
        # Salvar pipeline para transformações futuras
        self.pipeline = pipeline
        
        logger.info("Modelo de produtos treinado com sucesso")
    
    def recommend_similar_products(self, product_id: str, n_recommendations: int = 5) -> list:
        """
        Recomenda produtos similares ao produto especificado
        Args:
            product_id: ID do produto de referência (como string)
            n_recommendations: Número de recomendações a retornar
        Returns:
            Lista de IDs dos produtos recomendados (como strings)
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado corretamente")
        
        # Converter product_id para o tipo correto se necessário
        try:
            product_idx = np.where(self.product_ids == product_id)[0]
            if len(product_idx) == 0:
                raise ValueError(f"Produto com ID {product_id} não encontrado")
            
            product_idx = product_idx[0]
            
            # Obter features do produto
            product_features = self.pipeline.transform(
                self.product_features.iloc[product_idx:product_idx+1]
            )
            
            # Encontrar produtos similares
            distances, indices = self.model.kneighbors(
                product_features,
                n_neighbors=n_recommendations + 1  # +1 para excluir o próprio produto
            )
            
            # Excluir o próprio produto e obter IDs dos similares
            similar_indices = indices[0][1:]
            similar_product_ids = self.product_ids[similar_indices].tolist()
            
            return similar_product_ids
            
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {str(e)}")
            raise ValueError(f"Erro ao processar recomendações: {str(e)}")