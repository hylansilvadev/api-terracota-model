import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import logging
from datetime import datetime
import warnings

from nltk.corpus import stopwords
import nltk

from .database import database
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

nltk.download('stopwords')
portuguese_stopwords = stopwords.words('portuguese')

class AdvancedProductRecommender:
    def __init__(self):
        self.content_model = None
        self.collaborative_model = None
        self.hybrid_model = None
        self.matrix_factorization = None
        self.deep_features_model = None
        
        # Modelos híbridos
        self.ensemble_weights = {'content': 0.3, 'collaborative': 0.4, 'popularity': 0.2, 'temporal': 0.1}
        
        # Dados
        self.user_item_matrix = None
        self.user_profiles = None
        self.item_profiles = None
        self.popularity_scores = None
        self.temporal_weights = None
        
        # Mapas
        self.user_map = {}
        self.item_map = {}
        self.reverse_item_map = {}
        
        # Métricas de qualidade
        self.min_interactions_per_user = 2
        self.min_interactions_per_item = 2
        self.confidence_threshold = 0.4
        
        self._train_all_models()
    
    def _preprocess_data(self):
        """Preprocessamento avançado dos dados"""
        logger.info("Iniciando preprocessamento avançado...")
        
        # Simulando dados mais complexos baseados no seu schema
        with database.get_cursor() as cursor:
            # Query melhorada para incluir mais informações temporais e contextuais
            cursor.execute("""
                SELECT 
                    s.customer_id,
                    sp.products_ids,
                    COUNT(*) as purchase_count,
                    AVG(p.price) as avg_price,
                    STRING_AGG(DISTINCT p.type, ',') as product_types,
                    MAX(s.created_at) as last_purchase,
                    AVG(EXTRACT(EPOCH FROM (s.created_at - s.created_at::date))) as time_of_day,
                    COUNT(DISTINCT s.preference_id) as session_count
                FROM sales s
                JOIN sale_products sp ON s.preference_id = sp.sale_id
                JOIN products p ON sp.products_ids = p.id
                WHERE s.status = 'approved'
                GROUP BY s.customer_id, sp.products_ids
                HAVING COUNT(*) >= %s
            """, (self.min_interactions_per_user,))
            
            interactions = cursor.fetchall()
            
            # Buscar features dos produtos
            cursor.execute("""
                SELECT 
                    p.id,
                    p.name,
                    p.description,
                    p.price,
                    p.type,
                    p.quantity,
                    c.name as craftsman_name,
                    COUNT(sp.sale_id) as popularity,
                    AVG(CASE WHEN s.status = 'approved' THEN 5.0 ELSE 1.0 END) as implicit_rating,
                    EXTRACT(EPOCH FROM (NOW() - p.created_at))/86400 as days_since_created
                FROM products p
                LEFT JOIN craftsmen c ON p.craftsman_id = c.id
                LEFT JOIN sale_products sp ON p.id = sp.products_ids
                LEFT JOIN sales s ON sp.sale_id = s.preference_id
                GROUP BY p.id, p.name, p.description, p.price, p.type, p.quantity, c.name, p.created_at
                HAVING COUNT(sp.sale_id) >= %s
            """, (self.min_interactions_per_item,))
            
            products = cursor.fetchall()
        
        return interactions, products
    
    def _create_enhanced_features(self, products_df):
        """Criação de features avançadas"""
        logger.info("Criando features avançadas...")
        
        products_df['price'] = products_df['price'].astype(float)
        
        # Features de preço
        products_df['price_category'] = pd.cut(products_df['price'],
                                         bins=5,
                                         labels=['muito_barato', 'barato', 'medio', 'caro', 'muito_caro'])
        
        # Features temporais
        products_df['is_new'] = (products_df['days_since_created'] <= 30).astype(int)
        products_df['age_category'] = pd.cut(products_df['days_since_created'],
                                           bins=[0, 7, 30, 90, 365, float('inf')],
                                           labels=['nova', 'recente', 'medio', 'antigo', 'muito_antigo'])
        
        # Features de popularidade
        products_df['popularity_score'] = (products_df['popularity'] - products_df['popularity'].min()) / \
                                        (products_df['popularity'].max() - products_df['popularity'].min())
        
        # Features de texto mais sofisticadas
        products_df['text_features'] = products_df['name'].fillna('') + ' ' + \
                                     products_df['description'].fillna('') + ' ' + \
                                     products_df['craftsman_name'].fillna('')
        
        return products_df
    
    def _build_matrix_factorization(self, interactions_df):
        """Implementa Matrix Factorization (SVD) para capturar padrões latentes"""
        logger.info("Construindo modelo de Matrix Factorization...")
        
        # Criar matriz usuário-item com ratings implícitos
        user_ids = interactions_df['customer_id'].unique()
        item_ids = interactions_df['products_ids'].unique()
        
        self.user_map = {uid: i for i, uid in enumerate(user_ids)}
        self.item_map = {iid: i for i, iid in enumerate(item_ids)}
        self.reverse_item_map = {i: iid for iid, i in self.item_map.items()}
        
        # Criar ratings implícitos baseados em frequência e recência
        interactions_df['implicit_rating'] = np.log1p(interactions_df['purchase_count']) * 2
        
        # Adicionar peso temporal (compras mais recentes têm mais peso)
        if 'last_purchase' in interactions_df.columns:
            days_ago = (datetime.now() - pd.to_datetime(interactions_df['last_purchase'])).dt.days
            interactions_df['temporal_weight'] = np.exp(-days_ago / 365)  # Decaimento exponencial
            interactions_df['implicit_rating'] *= interactions_df['temporal_weight']
        
        # Criar matriz esparsa
        rows = interactions_df['customer_id'].map(self.user_map)
        cols = interactions_df['products_ids'].map(self.item_map)
        data = interactions_df['implicit_rating']
        
        self.user_item_matrix = csr_matrix((data, (rows, cols)), 
                                     shape=(len(user_ids), len(item_ids)))
        
        # Aplicar SVD
        self.matrix_factorization = TruncatedSVD(n_components=30, random_state=42)
        self.user_factors = self.matrix_factorization.fit_transform(self.user_item_matrix)
        self.item_factors = self.matrix_factorization.components_.T
        
        return self.user_factors, self.item_factors
        
    def _train_content_model(self, products_df):
        """Modelo de conteúdo aprimorado"""
        logger.info("Treinando modelo de conteúdo avançado...")
        
        # Pipeline mais sofisticado
        content_preprocessor = ColumnTransformer([
            ('text', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words=portuguese_stopwords
            ), 'text_features'),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), 
             ['type', 'craftsman_name', 'price_category', 'age_category']),
            ('numerical', StandardScaler(), ['price', 'popularity_score', 'days_since_created'])
        ])
        
        self.content_features = content_preprocessor.fit_transform(products_df)
        
        # Modelo KNN mais robusto
        self.content_model = NearestNeighbors(
            n_neighbors=20, 
            metric='cosine', 
            algorithm='brute'
        )
        self.content_model.fit(self.content_features)
        
        return self.content_model
    
    def _train_collaborative_model(self):
        """Modelo colaborativo baseado em Matrix Factorization"""
        logger.info("Treinando modelo colaborativo baseado em fatores latentes...")
        
        # Usar os fatores do SVD para encontrar usuários similares
        self.collaborative_model = NearestNeighbors(
            n_neighbors=15,
            metric='cosine',
            algorithm='brute'
        )
        self.collaborative_model.fit(self.user_factors)
        
        return self.collaborative_model
    
    def _calculate_confidence_scores(self, user_id, recommendations):
        """Calcula score de confiança para cada recomendação"""
        scores = []
        
        for item_id in recommendations:
            confidence = 0.0
            
            # Confiança baseada na popularidade do item
            if hasattr(self, 'item_popularity'):
                popularity = self.item_popularity.get(item_id, 0)
                confidence += 0.3 * min(popularity / 100, 1.0)  # Normalizado
            
            # Confiança baseada na densidade do usuário
            if user_id in self.user_map:
                user_interactions = self.user_item_matrix[self.user_map[user_id]].nnz
                confidence += 0.4 * min(user_interactions / 10, 1.0)  # Normalizado
            
            # Confiança baseada na qualidade do match
            confidence += 0.3  # Base score
            
            scores.append(confidence)
        
        return scores
    
    def _hybrid_recommendation(self, user_id, n_recommendations=10):
        """Sistema híbrido que combina múltiplas abordagens"""
        logger.info(f"Gerando recomendações híbridas para usuário {user_id}")
        
        recommendations = {}
        
        # 1. Recomendações colaborativas
        if user_id in self.user_map:
            collab_recs = self._get_collaborative_recommendations(user_id, n_recommendations * 2)
            for item_id, score in collab_recs:
                recommendations[item_id] = recommendations.get(item_id, 0) + \
                                         score * self.ensemble_weights['collaborative']
        
        # 2. Recomendações baseadas em conteúdo (dos itens que o usuário já comprou)
        user_items = self._get_user_items(user_id)
        if user_items:
            for user_item in user_items[:3]:  # Pegar os 3 itens mais recentes
                content_recs = self._get_content_recommendations(user_item, n_recommendations)
                for item_id, score in content_recs:
                    recommendations[item_id] = recommendations.get(item_id, 0) + \
                                             score * self.ensemble_weights['content']
        
        # 3. Recomendações por popularidade
        popular_recs = self._get_popular_recommendations(n_recommendations)
        for item_id, score in popular_recs:
            recommendations[item_id] = recommendations.get(item_id, 0) + \
                                     score * self.ensemble_weights['popularity']
        
        # 4. Filtrar itens já comprados pelo usuário
        user_purchased = set(user_items) if user_items else set()
        recommendations = {k: v for k, v in recommendations.items() if k not in user_purchased}
        
        # 5. Calcular confiança e filtrar
        final_recs = []
        for item_id, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
            confidence = self._calculate_item_confidence(user_id, item_id, score)
            if confidence >= self.confidence_threshold:
                final_recs.append((item_id, score, confidence))
            
            if len(final_recs) >= n_recommendations:
                break
        
        # Se não temos recomendações suficientes com alta confiança, relaxar o threshold
        if len(final_recs) < n_recommendations // 2:
            logger.warning(f"Poucas recomendações com alta confiança. Relaxando threshold.")
            final_recs = []
            for item_id, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]:
                confidence = self._calculate_item_confidence(user_id, item_id, score)
                final_recs.append((item_id, score, confidence))
        
        return final_recs[:n_recommendations]
    
    def _calculate_item_confidence(self, user_id, item_id, base_score):
        """Calcula confiança específica para um item"""
        confidence = base_score * 0.4  # Base score contribui 40%
        
        # Popularidade do item (30%)
        if hasattr(self, 'item_popularity') and item_id in self.item_popularity:
            pop_score = min(self.item_popularity[item_id] / 50, 1.0)  # Normalizado
            confidence += pop_score * 0.3
        
        # Qualidade do perfil do usuário (30%)
        if user_id in self.user_map:
            user_interactions = self.user_item_matrix[self.user_map[user_id]].nnz
            user_quality = min(user_interactions / 5, 1.0)  # Normalizado
            confidence += user_quality * 0.3
        
        return min(confidence, 1.0)
    
    def _get_collaborative_recommendations(self, user_id, n_recs):
        """Recomendações colaborativas usando Matrix Factorization"""
        if user_id not in self.user_map:
            return []
        
        user_idx = self.user_map[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Calcular similaridade com todos os itens
        item_scores = np.dot(self.item_factors, user_vector)
        
        # Remover itens já comprados
        purchased_items = self.user_item_matrix[user_idx].nonzero()[1]
        item_scores[purchased_items] = -np.inf
        
        # Pegar top items
        top_items = np.argsort(item_scores)[-n_recs:][::-1]
        
        recommendations = []
        for item_idx in top_items:
            if item_scores[item_idx] > 0:  # Apenas scores positivos
                item_id = self.reverse_item_map[item_idx]
                recommendations.append((item_id, item_scores[item_idx]))
        
        return recommendations
    
    def _get_content_recommendations(self, item_id, n_recs):
        """Recomendações baseadas em conteúdo"""
        # Implementar baseado no item_id
        # Retorna lista de tuplas (item_id, score)
        return []
    
    def _get_popular_recommendations(self, n_recs):
        """Recomendações por popularidade"""
        # Implementar baseado na popularidade
        return []
    
    def _get_user_items(self, user_id):
        """Busca itens que o usuário já comprou"""
        if user_id not in self.user_map:
            return []
        
        user_idx = self.user_map[user_id]
        purchased_items = self.user_item_matrix[user_idx].nonzero()[1]
        return [self.reverse_item_map[idx] for idx in purchased_items]
    
    def _train_all_models(self):
        """Treina todos os modelos"""
        try:
            # 1. Preprocessar dados
            interactions, products = self._preprocess_data()
            
            if not interactions or not products:
                logger.error("Dados insuficientes para treinar modelos")
                return
            
            # 2. Criar DataFrames
            interactions_df = pd.DataFrame(interactions, columns=[
                'customer_id', 'products_ids', 'purchase_count', 'avg_price',
                'product_types', 'last_purchase', 'time_of_day', 'session_count'
            ])
            
            products_df = pd.DataFrame(products, columns=[
                'id', 'name', 'description', 'price', 'type', 'quantity',
                'craftsman_name', 'popularity', 'implicit_rating', 'days_since_created'
            ])
            
            # 3. Criar features avançadas
            products_df = self._create_enhanced_features(products_df)
            
            # 4. Treinar Matrix Factorization
            self._build_matrix_factorization(interactions_df)
            
            # 5. Treinar modelo de conteúdo
            self._train_content_model(products_df)
            
            # 6. Treinar modelo colaborativo
            self._train_collaborative_model()
            
            # 7. Calcular métricas de popularidade
            self.item_popularity = products_df.set_index('id')['popularity'].to_dict()
            
            logger.info("Todos os modelos foram treinados com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro durante o treinamento: {e}")
            raise
    
    def evaluate_model_precision(self, test_size=0.2):
        """Avaliação rigorosa do modelo focada em precisão"""
        logger.info("Iniciando avaliação rigorosa do modelo...")
        
        # Criar conjunto de teste temporal
        all_interactions = []
        for user_id in self.user_map.keys():
            user_items = self._get_user_items(user_id)
            if len(user_items) >= 3:  # Usuários com pelo menos 3 interações
                # Últimos itens como teste
                test_items = user_items[-1:]
                train_items = user_items[:-1]
                all_interactions.append({
                    'user_id': user_id,
                    'train_items': train_items,
                    'test_items': test_items
                })
        
        if len(all_interactions) < 10:
            logger.warning("Poucos usuários para avaliação robusta")
            return
        
        # Métricas de avaliação
        precisions = []
        recalls = []
        confidences = []
        
        for interaction in all_interactions:
            user_id = interaction['user_id']
            true_items = set(interaction['test_items'])
            
            # Gerar recomendações
            recommendations = self._hybrid_recommendation(user_id, n_recommendations=10)
            
            if not recommendations:
                continue
            
            # Calcular métricas
            recommended_items = set([rec[0] for rec in recommendations])
            hits = len(recommended_items & true_items)
            
            precision = hits / len(recommended_items) if recommended_items else 0
            recall = hits / len(true_items) if true_items else 0
            avg_confidence = np.mean([rec[2] for rec in recommendations])
            
            precisions.append(precision)
            recalls.append(recall)
            confidences.append(avg_confidence)
        
        # Resultados
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_confidence = np.mean(confidences)
        
        logger.info(f"Resultados da Avaliação Rigorosa:")
        logger.info(f"Precisão Média: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
        logger.info(f"Recall Médio: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
        logger.info(f"Confiança Média: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        
        # Sugestões de melhoria
        if avg_precision < 0.9:
            logger.info("Sugestões para melhorar precisão:")
            logger.info("1. Aumentar threshold de confiança")
            logger.info("2. Coletar mais dados de feedback explícito")
            logger.info("3. Implementar filtros de qualidade mais rigorosos")
            logger.info("4. Usar deep learning para features mais sofisticadas")
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'confidence': avg_confidence,
            'n_evaluations': len(precisions)
        }
    
    def recommend_for_user(self, customer_id: str, n_recommendations: int = 5):
        """Interface principal para recomendações"""
        try:
            recommendations = self._hybrid_recommendation(customer_id, n_recommendations)
            
            # Retornar apenas IDs dos produtos com alta confiança
            high_confidence_recs = [
                rec[0] for rec in recommendations 
                if rec[2] >= self.confidence_threshold
            ]
            
            if len(high_confidence_recs) < n_recommendations // 2:
                logger.warning(f"Poucas recomendações de alta confiança para {customer_id}")
                # Retornar todas as recomendações disponíveis
                return [rec[0] for rec in recommendations]
            
            return high_confidence_recs
            
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações para {customer_id}: {e}")
            return self._get_popular_product_ids(n_recommendations)
    
    def _get_popular_product_ids(self, n: int) -> list:
        """Fallback para produtos populares"""
        if hasattr(self, 'item_popularity'):
            sorted_items = sorted(self.item_popularity.items(), 
                                key=lambda x: x[1], reverse=True)
            return [item[0] for item in sorted_items[:n]]
        return []
