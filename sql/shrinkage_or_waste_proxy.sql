-- Perishable Spoilage and Inventory Waste Proxy View
-- Analyzes demand volatility (coefficient of variation) across fresh/perishable departments
-- where order errors lead to physical waste and markdown write-offs rather than backorders.

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
    ROUND(mean_daily_sales * (std_daily_sales / NULLIF(mean_daily_sales, 0.0)), 2) AS spoilage_exposure_index,
    CASE
        WHEN perishability_class = 'PERISHABLE' AND (std_daily_sales / NULLIF(mean_daily_sales, 0.0)) >= 0.75
            THEN 'HIGH_SPOILAGE_RISK'
        WHEN perishability_class = 'PERISHABLE'
            THEN 'MODERATE_SPOILAGE_RISK'
        ELSE 'LOW_NON_PERISHABLE'
    END AS spoilage_risk_tier
FROM daily_stats
WHERE total_days >= 60
  AND mean_daily_sales >= 1.0
ORDER BY
    cv_demand DESC,
    mean_daily_sales DESC;
