import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import logging
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import nltk

# Configuração de logging e stopwords (mantido como no original)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
def stemmer_analyzer(text):
    stemmer = RSLPStemmer()
    return [stemmer.stem(word) for word in text.split()]

class ProductRecommender:
    def __init__(self):
        self.model = None
        self.pipeline = None
        self.train_product_ids = None
        
    
    def _train_model(self, products_df: pd.DataFrame):
        """Treina o modelo com o DataFrame fornecido, usando a feature 'text_feature'."""
        if products_df.empty:
            logger.warning("Nenhum produto fornecido para treinar o modelo")
            return

        logger.info(f"Treinando modelo com {len(products_df)} produtos...")
        
        self.train_product_ids = products_df['id'].values
        
        # --- MELHORIA APLICADA AQUI ---
        preprocessor = ColumnTransformer(
            transformers=[
                # O processador de texto agora usa a coluna 'text_feature'
                ('desc', TfidfVectorizer(
                    stop_words=stopwords.words('portuguese'),
                    #analyzer=stemmer_analyzer, # <-- Use o stemmer
                    ngram_range=(1, 2),        # <-- Use unigramas e bigramas
                    max_features=1500,         # <-- Aumentamos um pouco
                    min_df=1                   # <-- Reduzimos para 1
                ), 'text_feature'), # <-- APLICADO NA NOVA COLUNA COMBINADA
                
                ('type', OneHotEncoder(handle_unknown='ignore'), ['type']),
                ('price', StandardScaler(), ['price'])
            ],
            remainder='drop'
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
        ])
        
        # O fit_transform agora usa o DataFrame que já tem a coluna 'text_feature'
        features = pipeline.fit_transform(products_df)
        
        self.model = NearestNeighbors(
            n_neighbors=11,
            metric='cosine', # Cosine é excelente para dados de texto esparsos
            algorithm='brute'
        )
        self.model.fit(features)
        self.pipeline = pipeline
        
        logger.info("Modelo treinado com sucesso.")

    def recommend_similar_products(self, product_features_df: pd.DataFrame, n_recommendations: int = 5) -> list:
        # (código sem alterações)
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Chame _train_model() primeiro.")
        
        try:
            product_features_transformed = self.pipeline.transform(product_features_df)
            distances, indices = self.model.kneighbors(
                product_features_transformed,
                n_neighbors=n_recommendations
            )
            similar_indices = indices[0]
            similar_product_ids = self.train_product_ids[similar_indices].tolist()
            return similar_product_ids
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {str(e)}")
            raise ValueError(f"Erro ao processar recomendações: {str(e)}")

    def evaluate(self, all_products_df: pd.DataFrame, test_size=0.2, k=5):
        """
        Divide os dados, treina o modelo e avalia a precisão das recomendações.
        A "precisão" é baseada na correspondência de 'type'.
        """
        logger.info("Iniciando avaliação do modelo de recomendação...")

        # 1. Dividir os dados
        # Usamos stratify='type' para garantir que a proporção de cada tipo de produto
        # seja a mesma nos conjuntos de treino e teste. Isso é crucial para uma boa avaliação.
        try:
            train_df, test_df = train_test_split(
                all_products_df, 
                test_size=test_size, 
                random_state=42, 
                stratify=all_products_df['type']
            )
        except ValueError:
            # Fallback caso a estratificação falhe (poucas amostras de um tipo)
            train_df, test_df = train_test_split(
                all_products_df, 
                test_size=test_size, 
                random_state=42
            )

        # 2. Treinar o modelo apenas com os dados de treino
        self._train_model(train_df)

        # 3. Avaliar
        total_precision = 0
        test_count = 0

        for _, test_product in test_df.iterrows():
            # Pega as features do produto de teste em um formato de DataFrame
            test_product_features = test_product.to_frame().T
            
            # Pega o tipo do produto de teste (nosso "gabarito")
            true_type = test_product['type']
            
            # Gera recomendações
            try:
                recommended_ids = self.recommend_similar_products(test_product_features, n_recommendations=k)
                
                # Busca os tipos dos produtos recomendados
                recommended_products = train_df[train_df['id'].isin(recommended_ids)]
                recommended_types = recommended_products['type'].tolist()
                
                # Calcula a precisão para esta recomendação
                # Conta quantos dos recomendados têm o mesmo tipo do produto de teste
                correct_recommendations = sum(1 for r_type in recommended_types if r_type == true_type)
                precision = correct_recommendations / k
                total_precision += precision
                test_count += 1

            except Exception as e:
                logger.warning(f"Não foi possível gerar recomendação para o produto de teste {test_product['id']}: {e}")

        # Calcula a média da Precision@k em todos os produtos de teste
        average_precision_at_k = total_precision / test_count if test_count > 0 else 0
        
        logger.info(f"--- Resultados da Avaliação ---")
        logger.info(f"Tamanho do Conjunto de Treino: {len(train_df)}")
        logger.info(f"Tamanho do Conjunto de Teste: {len(test_df)}")
        logger.info(f"Precisão Média @{k} (baseado no tipo): {average_precision_at_k:.2%}")
        logger.info(f"-------------------------------")

        return average_precision_at_k