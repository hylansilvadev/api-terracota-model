# Sistema de Recomendação de Produtos Artesanais 🚀
API RESTful construída com FastAPI para fornecer recomendações inteligentes de produtos artesanais, utilizando um modelo de Machine Learning baseado em conteúdo.

## 📜 Descrição
Este projeto tem como objetivo oferecer uma experiência de descoberta de produtos aprimorada em uma plataforma de e-commerce de artesanato. Através de um endpoint de recomendação, a API é capaz de sugerir produtos similares a um item de referência, auxiliando os usuários a encontrar itens relevantes e aumentando o engajamento na plataforma.

O sistema é construído com foco em performance e escalabilidade, utilizando o framework assíncrono FastAPI e bibliotecas de Data Science consolidadas como Scikit-learn e Pandas.

# ✨ Features
API de Produtos: Endpoint para listar todos os produtos disponíveis.
Recomendações Baseadas em Conteúdo: Endpoint que recebe o ID de um produto e retorna uma lista de itens similares.
Treinamento na Inicialização: O modelo de recomendação é treinado e avaliado automaticamente quando a API é iniciada, garantindo que ele esteja sempre pronto e atualizado com os dados mais recentes do banco.
Avaliação de Performance: A precisão do modelo (Precision@k) é calculada e exibida nos logs na inicialização, permitindo o monitoramento contínuo da sua qualidade.

# 🧠 O Modelo de Recomendação
O coração deste projeto é um modelo de recomendação que utiliza a abordagem de Filtros Baseados em Conteúdo (Content-Based Filtering). A premissa é simples: "Se você gostou deste item, provavelmente gostará de outros itens com características parecidas."

## Como Funciona?
O processo para gerar recomendações é dividido em três etapas principais:

Extração de Features: Para cada produto, extraímos um conjunto de características do banco de dados que o descrevem. Atualmente, utilizamos:

Dados Textuais: nome e descrição do produto.
Dados Categóricos: type (categoria do produto).
Dados Numéricos: price (preço).
Pré-processamento e Vetorização: Os dados brutos não podem ser diretamente utilizados pelo modelo. Eles passam por um pipeline de transformação:

Texto (text_feature): O nome e a descrição são combinados em um único campo. Em seguida, um TfidfVectorizer transforma este texto em um vetor numérico. Este processo considera a relevância das palavras, remove stopwords (palavras comuns como "o", "a", "de") e utiliza N-gramas para capturar pequenas frases (ex: "potes de barro").
Preço (price): O preço é normalizado com StandardScaler para que sua escala não domine as outras features.
Tipo (type): A categoria é transformada em um formato numérico usando OneHotEncoder.
Cálculo de Similaridade (KNN): Com todos os produtos representados como vetores numéricos, utilizamos o algoritmo NearestNeighbors (baseado em k-NN) com a métrica de similaridade de cosseno. Ele calcula a "distância" entre o vetor do produto de referência e os vetores de todos os outros produtos no catálogo. Os produtos com a menor distância (ou seja, os vetores mais "próximos") são considerados os mais similares e são retornados como recomendação.

Opcionalmente, o modelo pode ser estendido para usar Embeddings de Texto (BERT) para uma compreensão semântica mais profunda, oferecendo um caminho claro para futuras melhorias de precisão.

# 🛠️ Tecnologias Utilizadas
- Backend: FastAPI, Uvicorn
- Data Science e ML: Scikit-learn, Pandas, NumPy, NLTK
- Banco de Dados: PostgreSQL
- Validação de Dados: Pydantic

# ⚙️ Instalação e Configuração
Siga os passos abaixo para executar o projeto localmente.

## Pré-requisitos
- Python 3.9 ou superior
- Um gerenciador de pacotes como pip
### Passos
1. Clone o repositório:

```
Bash

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
Crie e ative um ambiente virtual (recomendado):

```

``` Bash

python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
Instale as dependências:
```
```
Bash

pip install -r requirements.txt
Configure as variáveis de ambiente:
Crie um arquivo .env na raiz do projeto, baseado no arquivo .env.example (se houver). Adicione as credenciais de acesso ao seu banco de dados.
```
```
Snippet de código

# Exemplo de .env
DATABASE_URL="postgresql://user:password@host:port/database"
Execute o download de recursos do NLTK (necessário na primeira vez):
Execute o seguinte comando no terminal Python:
```
```
Python

import nltk
nltk.download('stopwords')
Inicie a API:
```
```
Bash

uvicorn app.main:app --reload
A API estará disponível em http://127.0.0.1:8000.
```

# 📦 Uso da API (Endpoints)
A documentação interativa (Swagger UI) está disponível em http://127.0.0.1:8000/docs.

1. Listar todos os produtos
 - Endpoint: GET /products/
 - Descrição: Retorna uma lista de todos os produtos cadastrados no sistema.
- Resposta (200 OK):
```
JSON

[
  {
    "id": "e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0",
    "name": "Bijuteria de Resina",
    "description": "Colar feito com resina colorida.",
    "price": 95.0,
    "type": "RESIN_ART",
    "photo_url": "http://example.com/photo.jpg",
    "craftsman_name": "Artesã Maria"
  }
]

```
2. Obter Recomendações de Produtos
 - Endpoint: POST /recommend/products/
 - Descrição: Recomenda produtos similares com base no ID de um produto de referência.
 - Corpo da Requisição (Request Body):
```
JSON

{
  "product_id": "e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b0",
  "n_recommendations": 5
}
Resposta (200 OK):

```
```
JSON

{
  "recommended_products": [
    {
      "id": "f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6",
      "nome": "Brinco de Resina Abstrato",
      "descricao": "Brinco leve feito com resina e pigmentos.",
      "preco": 75.5,
      "estoque": 15,
      "imagemUrl": "http://example.com/brinco.jpg",
      "status": "ativo",
      "categoria": "RESIN_ART",
      "totalVendas": 0
    }
  ]
}
```