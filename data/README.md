# Data

This project uses the Kaggle competition
[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
(Corporación Favorita, Ecuador).

> Raw files are **never committed**. `data/raw/` and `data/interim/` are listed in
> `.gitignore`. Before using the data, read and accept the competition rules on
> Kaggle and check that your intended use is permitted.

## Expected files

Place the following files, unmodified, in `data/raw/`:

| File | Description |
| --- | --- |
| `train.csv` | Daily unit sales by store and product family, with `onpromotion` |
| `test.csv` | 16-day forecast period (store, family, date, onpromotion) |
| `stores.csv` | Store metadata: city, state, type, cluster |
| `oil.csv` | Daily WTI oil price |
| `holidays_events.csv` | Holidays and events with transfer flags |
| `transactions.csv` | Daily transaction counts per store |
| `sample_submission.csv` | Submission format for the test period |

## Download (PowerShell)

1. Install the Kaggle CLI in the project virtual environment:

   ```powershell
   python -m pip install kaggle
   ```

2. Create an API token on kaggle.com (Account > API > Create New Token) and save
   it as `%USERPROFILE%\.kaggle\kaggle.json`. Never place it inside the repository.

3. Accept the competition rules on the competition page, then run from the
   repository root:

   ```powershell
   kaggle competitions download -c store-sales-time-series-forecasting -p data/raw
   Expand-Archive data/raw/store-sales-time-series-forecasting.zip -DestinationPath data/raw
   Remove-Item data/raw/store-sales-time-series-forecasting.zip
   ```

Alternatively, download the zip from the competition's Data tab in a browser and
extract it into `data/raw/`.
