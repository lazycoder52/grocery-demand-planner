-- Promotional Sales Lift and Elasticity Analysis by Store and Product Family
-- Filters data to post 2014-04-01 when promotion tracking began in Corporación Favorita.
-- Compares sales velocity on promotional days vs non-promotional baseline days.

WITH store_opening AS (
    SELECT
        store_nbr,
        MIN(date) AS first_sale_date
    FROM train
    WHERE sales > 0
    GROUP BY store_nbr
),

base_data AS (
    SELECT
        t.store_nbr,
        s.city,
        s.state,
        s.type AS store_type,
        s.cluster AS store_cluster,
        t.family,
        t.date,
        t.sales,
        t.onpromotion,
        CASE WHEN t.onpromotion > 0 THEN 1 ELSE 0 END AS is_promo
    FROM train t
    JOIN stores s
        ON t.store_nbr = s.store_nbr
    JOIN store_opening so
        ON t.store_nbr = so.store_nbr
    WHERE t.date >= DATE '2014-04-01'
      AND t.date >= so.first_sale_date
      AND NOT (EXTRACT(month FROM t.date) = 12 AND EXTRACT(day FROM t.date) = 25)
),

store_family_agg AS (
    SELECT
        store_nbr,
        city,
        state,
        store_type,
        store_cluster,
        family,
        CAST(COUNT(*) AS INTEGER) AS total_evaluated_days,
        CAST(SUM(is_promo) AS INTEGER) AS promo_days,
        CAST(SUM(1 - is_promo) AS INTEGER) AS non_promo_days,
        ROUND(100.0 * SUM(is_promo) / COUNT(*), 2) AS promo_day_share_pct,
        ROUND(AVG(CASE WHEN is_promo = 1 THEN sales ELSE NULL END), 2) AS promo_mean_daily_sales,
        ROUND(AVG(CASE WHEN is_promo = 0 THEN sales ELSE NULL END), 2) AS non_promo_mean_daily_sales,
        ROUND(SUM(CASE WHEN is_promo = 1 THEN sales ELSE 0.0 END), 2) AS promo_total_sales,
        ROUND(SUM(sales), 2) AS total_sales
    FROM base_data
    GROUP BY
        store_nbr,
        city,
        state,
        store_type,
        store_cluster,
        family
)

SELECT
    store_nbr,
    city,
    state,
    store_type,
    store_cluster,
    family,
    total_evaluated_days,
    promo_days,
    non_promo_days,
    promo_day_share_pct,
    promo_mean_daily_sales,
    non_promo_mean_daily_sales,
    promo_total_sales,
    total_sales,
    ROUND(
        100.0 * (promo_mean_daily_sales - non_promo_mean_daily_sales)
        / NULLIF(non_promo_mean_daily_sales, 0.0),
        2
    ) AS empirical_promo_lift_pct
FROM store_family_agg
WHERE non_promo_days >= 30
  AND promo_days >= 10
  AND non_promo_mean_daily_sales > 0.1
ORDER BY
    empirical_promo_lift_pct DESC;
