-- Clean weekly analytical view joining sales, oil, transactions, and holiday indicators.
-- Truncates daily dates to Monday-based week start (date_trunc('week', date)).

WITH weekly_sales AS (
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
),

weekly_transactions AS (
    SELECT
        store_nbr,
        CAST(date_trunc('week', date) AS DATE) AS week_start_date,
        CAST(SUM(transactions) AS INTEGER) AS weekly_transactions,
        ROUND(AVG(transactions), 2) AS mean_daily_transactions
    FROM transactions
    GROUP BY
        store_nbr,
        date_trunc('week', date)
),

weekly_oil AS (
    SELECT
        CAST(date_trunc('week', date) AS DATE) AS week_start_date,
        ROUND(AVG(dcoilwtico), 2) AS avg_oil_price,
        CAST(COUNT(dcoilwtico) AS INTEGER) AS oil_trading_days
    FROM oil
    GROUP BY
        date_trunc('week', date)
),

weekly_holidays AS (
    SELECT
        CAST(date_trunc('week', date) AS DATE) AS week_start_date,
        CAST(COUNT(*) AS INTEGER) AS holiday_count,
        CAST(SUM(CASE WHEN transferred = FALSE THEN 1 ELSE 0 END) AS INTEGER) AS non_transferred_holidays
    FROM holidays_events
    GROUP BY
        date_trunc('week', date)
)

SELECT
    ws.store_nbr,
    st.city,
    st.state,
    st.type AS store_type,
    st.cluster AS store_cluster,
    ws.family,
    ws.week_start_date,
    ws.total_sales,
    ws.mean_daily_sales,
    ws.promo_days,
    ws.total_promo_count,
    ws.active_days,
    COALESCE(wt.weekly_transactions, 0) AS weekly_transactions,
    wo.avg_oil_price,
    COALESCE(wh.holiday_count, 0) AS holiday_count,
    COALESCE(wh.non_transferred_holidays, 0) AS non_transferred_holidays
FROM weekly_sales ws
LEFT JOIN stores st
    ON ws.store_nbr = st.store_nbr
LEFT JOIN weekly_transactions wt
    ON ws.store_nbr = wt.store_nbr
    AND ws.week_start_date = wt.week_start_date
LEFT JOIN weekly_oil wo
    ON ws.week_start_date = wo.week_start_date
LEFT JOIN weekly_holidays wh
    ON ws.week_start_date = wh.week_start_date
ORDER BY
    ws.store_nbr,
    ws.family,
    ws.week_start_date;
