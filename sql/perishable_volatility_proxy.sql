-- Perishable Demand Volatility Diagnostic View (Proxy for Perishable Inventory Risk)
-- Analyzes demand volatility (coefficient of variation, CV = std / mean) across fresh/perishable departments.
-- IMPORTANT: Physical spoilage, scrap, and shrinkage logs are not recorded in the dataset.
-- High coefficient of variation in perishables serves as an empirical proxy for inventory volatility exposure,
-- not an audit of confirmed physical shrinkage.

WITH daily_stats AS (
    SELECT
        t.store_nbr,
        s.city,
        s.type AS store_type,
        t.family,
        CASE
            WHEN t.family IN (
                'DAIRY', 'MEATS', 'POULTRY', 'PRODUCE',
                'SEAFOOD', 'BREAD/BAKERY', 'DELI', 'EGGS'
            ) THEN 'PERISHABLE'
            ELSE 'NON_PERISHABLE'
        END AS perishability_class,
        CAST(COUNT(*) AS INTEGER) AS total_days,
        ROUND(AVG(t.sales), 2) AS mean_daily_sales,
        ROUND(STDDEV_POP(t.sales), 2) AS std_daily_sales,
        ROUND(MAX(t.sales), 2) AS max_daily_sales,
        CAST(SUM(CASE WHEN t.sales = 0 THEN 1 ELSE 0 END) AS INTEGER) AS zero_sales_days
    FROM train t
    JOIN stores s
        ON t.store_nbr = s.store_nbr
    GROUP BY
        t.store_nbr,
        s.city,
        s.type,
        t.family
)

SELECT
    store_nbr,
    city,
    store_type,
    family,
    perishability_class,
    total_days,
    mean_daily_sales,
    std_daily_sales,
    ROUND(std_daily_sales / NULLIF(mean_daily_sales, 0.0), 4) AS cv_demand,
    max_daily_sales,
    zero_sales_days,
    ROUND(mean_daily_sales * (std_daily_sales / NULLIF(mean_daily_sales, 0.0)), 2) AS volatility_exposure_index,
    CASE
        WHEN perishability_class = 'PERISHABLE' AND (std_daily_sales / NULLIF(mean_daily_sales, 0.0)) >= 0.75
            THEN 'HIGH_VOLATILITY_PERISHABLE_PROXY'
        WHEN perishability_class = 'PERISHABLE'
            THEN 'MODERATE_VOLATILITY_PERISHABLE_PROXY'
        ELSE 'NON_PERISHABLE_OR_LOW_VOLATILITY'
    END AS volatility_proxy_tier
FROM daily_stats
WHERE total_days >= 60
  AND mean_daily_sales >= 1.0
ORDER BY
    cv_demand DESC,
    mean_daily_sales DESC;
