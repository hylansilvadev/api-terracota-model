QUERIES = {
    # CTE que une as informações principais do produto
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

    # Query para buscar todos os produtos
    "get_all_products": """
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
        SELECT * FROM product_details
    """,
    
    # Query para buscar produtos específicos por ID
    "get_similar_products": """
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
        SELECT * FROM product_details WHERE id = ANY(%s)
    """,
    
    # Query para buscar os produtos mais populares
    "get_popular_products": """
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
        LIMIT %s
    """,

    # Query para features básicas
    "get_product_features": """
        SELECT 
            id,
            name,
            description,
            price,
            type
        FROM 
            products
    """,
    
    # Query específica para buscar um produto por ID (CORRIGIDA COM TEXT_FEATURE)
    "get_product_by_id": """
        SELECT 
            id,
            name,
            description,
            price,
            type,
            COALESCE(name, '') || ' ' || COALESCE(description, '') || ' ' || COALESCE(type, '') as text_feature
        FROM 
            products
        WHERE id = %s
    """
}