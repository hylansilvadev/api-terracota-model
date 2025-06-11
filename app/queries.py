QUERIES = {
    "product_details_cte": """
        WITH product_details AS (
            SELECT 
                p.id, p.name, p.description, p.price, p.quantity, p.type,
                p.created_at, p.updated_at, r.location AS photo_url, c.name AS craftsman_name
            FROM products p
            LEFT JOIN resources r ON p.photo_id = r.id
            LEFT JOIN craftsmen c ON p.craftsman_id = c.id
        )
    """,
    "get_all_products": """
        {product_details_cte}
        SELECT * FROM product_details;
    """,
    "get_similar_products": """
        {product_details_cte}
        SELECT * FROM product_details WHERE id = ANY(%s::uuid[]);
    """,
    "get_popular_products": """
        {product_details_cte}
        SELECT pd.*, COUNT(sp.sale_id) AS purchase_count
        FROM product_details pd
        JOIN sale_products sp ON pd.id = sp.products_ids -- Corrigido de sp.product_id
        GROUP BY pd.id, pd.name, pd.description, pd.price, pd.quantity, 
            pd.type, pd.created_at, pd.updated_at, pd.photo_url, pd.craftsman_name
        ORDER BY purchase_count DESC
        LIMIT %s;
    """,
    "get_user_purchase_history": """
        SELECT 
            s.customer_id, 
            sp.products_ids,
            COUNT(*) as purchase_count
        FROM sales s
        JOIN sale_products sp ON s.preference_id = sp.sale_id
        WHERE s.status = 'approved'
        GROUP BY s.customer_id, sp.products_ids;
    """,
    "get_product_features": """
        SELECT id, name, description, price, type
        FROM products;
    """,
    "get_product_features": """
        SELECT
            p.id,
            p.name,
            p.description,
            p.price,
            p.type,
            c.name AS craftsman_name
        FROM
            products p
        LEFT JOIN
            craftsmen c ON p.craftsman_id = c.id;
    """,
    "get_user_purchase_history": """
        SELECT 
            s.customer_id, 
            sp.products_ids,
            COUNT(*) as purchase_count
        FROM sales s
        JOIN sale_products sp ON s.preference_id = sp.sale_id
        WHERE s.status = 'approved'
        GROUP BY s.customer_id, sp.products_ids;
    """,
    "get_popular_products": """
        SELECT 
            p.id
        FROM 
            products p
        JOIN 
            sale_products sp ON p.id = sp.products_ids
        GROUP BY 
            p.id
        ORDER BY 
            COUNT(sp.sale_id) DESC
        LIMIT %s;
    """,
    "get_similar_products": """
        SELECT 
            p.id, p.name, p.description, p.price, p.quantity, 
            p.type, r.location AS photo_url
        FROM products p
        LEFT JOIN resources r ON p.photo_id = r.id
        WHERE p.id = ANY(CAST(%s AS char(32)[]));
    """,
}