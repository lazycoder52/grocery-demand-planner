-- Schema definition for Corporación Favorita demand planning tables in DuckDB

CREATE TABLE IF NOT EXISTS stores (
    store_nbr INTEGER PRIMARY KEY,
    city VARCHAR NOT NULL,
    state VARCHAR NOT NULL,
    type VARCHAR NOT NULL,
    cluster INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS oil (
    date DATE PRIMARY KEY,
    dcoilwtico DOUBLE
);

CREATE TABLE IF NOT EXISTS holidays_events (
    date DATE NOT NULL,
    type VARCHAR NOT NULL,
    locale VARCHAR NOT NULL,
    locale_name VARCHAR NOT NULL,
    description VARCHAR NOT NULL,
    transferred BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    date DATE NOT NULL,
    store_nbr INTEGER NOT NULL,
    transactions INTEGER NOT NULL,
    PRIMARY KEY (date, store_nbr)
);

CREATE TABLE IF NOT EXISTS train (
    id BIGINT PRIMARY KEY,
    date DATE NOT NULL,
    store_nbr INTEGER NOT NULL,
    family VARCHAR NOT NULL,
    sales DOUBLE NOT NULL,
    onpromotion INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS test (
    id BIGINT PRIMARY KEY,
    date DATE NOT NULL,
    store_nbr INTEGER NOT NULL,
    family VARCHAR NOT NULL,
    onpromotion INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sample_submission (
    id BIGINT PRIMARY KEY,
    sales DOUBLE NOT NULL
);
