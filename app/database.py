import os
import psycopg2
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()

class Database:
    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.port = os.getenv("DB_PORT")
        self.name = os.getenv("DB_NAME")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        self.sslmode = os.getenv("DB_SSLMODE")
    
    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.name,
                user=self.user,
                password=self.password,
                sslmode='require'
            )
            yield conn
        except Exception as e:
            print(f"Erro ao conectar ao banco de dados: {e}")
            raise
        finally:
            if conn is not None:
                conn.close()
    
    @contextmanager
    def get_cursor(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Erro na transação: {e}")
                raise
            finally:
                cursor.close()

database = Database()