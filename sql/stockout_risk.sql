-- Stockout Risk and Censored Demand Diagnostic View
-- Identifies high-velocity store-family items experiencing elevated zero-sales days,
-- which frequently represents on-shelf availability failure rather than zero true demand.

WITH series_stats AS (
    SELECT
        t.store_nbr,
        s.city,
        s.state,
        s.type AS store_type,
        t.family,
        CAST(COUNT(*) AS INTEGER) AS total_days,
        CAST(SUM(CASE WHEN t.sales = 0 THEN 1 ELSE 0 END) AS INTEGER) AS zero_sales_days,
        ROUND(AVG(t.sales), 2) AS mean_daily_sales,
        ROUND(AVG(CASE WHEN t.sales > 0 THEN t.sales ELSE NULL END), 2) AS active_mean_daily_sales,
        ROUND(MAX(t.sales), 2) AS max_daily_sales,
        ROUND(SUM(t.sales), 2) AS total_sales
    FROM train t
    JOIN stores s
        ON t.store_nbr = s.store_nbr
    GROUP BY
        t.store_nbr,
        s.city,
        s.state,
        s.type,
        t.family
)

SELECT
    store_nbr,
    city,
    state,
    store_type,
    family,
    total_days,
    zero_sales_days,
    ROUND(100.0 * zero_sales_days / total_days, 2) AS zero_sales_rate_pct,
    mean_daily_sales,
    active_mean_daily_sales,
    max_daily_sales,
    total_sales,
    CASE
        WHEN mean_daily_sales >= 50.0 AND (1.0 * zero_sales_days / total_days) >= 0.10
            THEN 'CRITICAL_HIGH_VELOCITY_STOCKOUT_RISK'
        WHEN mean_daily_sales >= 15.0 AND (1.0 * zero_sales_days / total_days) >= 0.25
            THEN 'MODERATE_STOCKOUT_RISK'
        WHEN mean_daily_sales < 5.0 AND (1.0 * zero_sales_days / total_days) >= 0.50
            THEN 'SLOW_MOVING_INTERMITTENT_TAIL'
        ELSE 'STABLE_IN_STOCK'
    END AS stockout_risk_segment
FROM series_stats
WHERE total_days >= 60
ORDER BY
    mean_daily_sales DESC,
    zero_sales_rate_pct DESC;
