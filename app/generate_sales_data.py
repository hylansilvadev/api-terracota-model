import os
import psycopg2
import uuid
from faker import Faker
from dotenv import load_dotenv
from contextlib import contextmanager
import random
from datetime import datetime
from collections import defaultdict

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configurações ---
# Defina quantas novas vendas você quer criar
NUM_SALES = 500
# ---------------------

fake = Faker('pt_BR')

class Database:
    """ Classe de conexão com o banco """
    def __init__(self):
        self.conn_str = "host={host} port={port} dbname={name} user={user} password={password} sslmode={sslmode}".format(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            name=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            sslmode=os.getenv("DB_SSLMODE", "require")
        )

    @contextmanager
    def get_cursor(self):
        conn = psycopg2.connect(self.conn_str)
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

db = Database()

def generate_preference_id():
    """ Gera um preference_id no formato visto nos dados reais """
    p1 = random.randint(100000000, 999999999)
    p2 = uuid.uuid4()
    return f"{p1}-{p2}"

def generate_payment_id():
    """ Gera um payment_id NUMÉRICO e grande, compatível com bigint. """
    return random.randint(100000000000, 999999999999)

def create_sales(customer_ids, products_data):
    """
    Cria e insere vendas sintéticas usando clientes e produtos existentes.
    'products_data' deve ser um dicionário: {craftsman_id: [lista_de_seus_produtos]}
    """
    print(f"Iniciando a criação de {NUM_SALES} novas vendas...")
    sales_to_insert = []
    sale_products_map = []

    all_craftsmen_ids = list(products_data.keys())

    for _ in range(NUM_SALES):
        # 1. Escolhe um cliente aleatório
        customer_id = random.choice(customer_ids)
        
        # 2. Simula uma "visita à loja": escolhe um artesão
        craftsman_id = random.choice(all_craftsmen_ids)
        
        # 3. Pega os produtos disponíveis para esse artesão
        available_products = products_data[craftsman_id]
        if not available_products:
            continue
            
        # 4. Monta um "carrinho de compras"
        num_items_in_cart = random.randint(1, min(len(available_products), 4))
        cart = random.sample(available_products, num_items_in_cart)
        
        # 5. Calcula o total e prepara os dados
        total_price = sum(item['price'] for item in cart)
        preference_id = generate_preference_id()
        payment_id = generate_payment_id()
        
        # 6. Monta a tupla para a tabela 'sales'
        sale_tuple = (
            preference_id,
            craftsman_id,
            customer_id,
            random.choice(['CREDIT_CARD', 'PIX', 'BOLETO']),
            round(total_price, 2),
            fake.past_datetime(start_date="-1y"),
            payment_id,
            'approved'  # Usando o status correto!
        )
        sales_to_insert.append(sale_tuple)

        # 7. Mapeia os produtos para a venda na tabela 'sale_products'
        for item in cart:
            sale_products_map.append((preference_id, item['id']))

    if not sales_to_insert:
        print("Nenhuma venda válida foi gerada.")
        return

    # 8. Insere tudo no banco
    with db.get_cursor() as cur:
        cur.executemany("""
            INSERT INTO sales (preference_id, craftsman_id, customer_id, payment_method, total, created_at, payment_id, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, sales_to_insert)
        
        cur.executemany("""
            INSERT INTO sale_products (sale_id, products_ids)
            VALUES (%s, %s)
        """, sale_products_map)

    print(f"SUCESSO: {len(sales_to_insert)} novas vendas foram adicionadas ao banco de dados.")

def main():
    """ Ponto de entrada do script. """
    print("--- Script para Adicionar Histórico de Compras Sintético ---")
    try:
        with db.get_cursor() as cur:
            # Pega todos os clientes existentes
            cur.execute("SELECT id FROM customers")
            all_customer_ids = [row[0] for row in cur.fetchall()]
            
            # Pega todos os produtos e organiza por artesão
            cur.execute("SELECT id, craftsman_id, price FROM products")
            products_by_craftsman = defaultdict(list)
            for prod_id, craft_id, price in cur.fetchall():
                if craft_id: # Ignora produtos sem artesão
                    products_by_craftsman[craft_id].append({'id': prod_id, 'price': float(price)})
        
        if not all_customer_ids or not products_by_craftsman:
            print("ERRO: Não foram encontrados clientes ou produtos suficientes no banco. Canceleando.")
            return
            
        print(f"Encontrados {len(all_customer_ids)} clientes e produtos de {len(products_by_craftsman)} artesãos.")
        create_sales(all_customer_ids, products_by_craftsman)

    except (psycopg2.Error, Exception) as e:
        print(f"Ocorreu um erro de banco de dados: {e}")

if __name__ == "__main__":
    main()