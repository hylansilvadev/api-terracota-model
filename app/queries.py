QUERIES = {
    # CTE que une as informações principais do produto para ser reutilizada.
    # Isso evita a repetição dos mesmos JOINs em várias consultas.
    "product_details_cte": """
        WITH product_details AS (
            SELECT 
                p.id,
                p.name,
                p.description,
                p.price,
                p.quantity,
                p.type,
                p.created_at,
                p.updated_at,
                r.location AS photo_url,
                c.name AS craftsman_name
            FROM 
                products p
            LEFT JOIN 
                resources r ON p.photo_id = r.id
            LEFT JOIN 
                craftsmen c ON p.craftsman_id = c.id
        )
    """,

    # Query para buscar todos os produtos usando a CTE.
    "get_all_products": """
        {product_details_cte}
        SELECT * FROM product_details;
    """,
    
    # Query para buscar produtos específicos por ID usando a CTE.
    "get_similar_products": """
        {product_details_cte}
        SELECT * FROM product_details WHERE id = ANY(%s);
    """,
    
    # Query para buscar os produtos mais populares, também usando a CTE.
    "get_popular_products": """
        {product_details_cte}
        SELECT 
            pd.*,
            COUNT(sp.sale_id) AS purchase_count
        FROM 
            product_details pd
        JOIN 
            sale_products sp ON pd.id = sp.product_id
        GROUP BY 
            pd.id, pd.name, pd.description, pd.price, pd.quantity, 
            pd.type, pd.created_at, pd.updated_at, pd.photo_url, pd.craftsman_name
        ORDER BY 
            purchase_count DESC
        LIMIT %s;
    """,

    # Query simples que não necessita da complexidade dos JOINs da CTE.
    "get_product_features": """
        SELECT 
            id,
            name,
            description,
            price,
            type
        FROM 
            products;
    """
}

# Exemplo de como você usaria isso em seu código Python, formatando a string:
# cursor.execute(QUERIES["get_all_products"].format(product_details_cte=QUERIES["product_details_cte"]))