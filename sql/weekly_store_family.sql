-- Weekly aggregation of store sales by store_nbr and product family
-- Truncates daily dates to Monday-based week start (date_trunc('week', date))
SELECT
    store_nbr,
    family,
    CAST(date_trunc('week', date) AS DATE) AS week_start_date,
    ROUND(SUM(sales), 2) AS total_sales,
    ROUND(AVG(sales), 2) AS mean_daily_sales,
    CAST(SUM(CASE WHEN onpromotion > 0 THEN 1 ELSE 0 END) AS INTEGER) AS promo_days,
    CAST(SUM(onpromotion) AS INTEGER) AS total_promo_count,
    CAST(COUNT(*) AS INTEGER) AS active_days
FROM train
GROUP BY
    store_nbr,
    family,
    date_trunc('week', date)
ORDER BY
    store_nbr,
    family,
    week_start_date;
