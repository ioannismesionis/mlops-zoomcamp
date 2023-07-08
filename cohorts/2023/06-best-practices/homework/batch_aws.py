#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import click


def read_data(filename, categorical):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def prepare_data(df, year, month, categorical):
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.mean())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    return df_result


# @click.command()
# @click.option("--year", default=2022, help="Define the year")
# @click.option("--month", default=2, help="Define the month")
def main(year, month, categorical=["PULocationID", "DOLocationID"]):
    INPUT_FILE = "taxi_type_test_data=yellow_year=2022_month=02.parquet"
    OUTPUT_FILE = f"taxi_type_aws=yellow_year={year:04d}_month={month:02d}.parquet"

    df = read_data(INPUT_FILE, categorical)

    df_result = prepare_data(df, year, month, categorical)

    df_result.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)

    print("Total sum:", df_result["predicted_duration"].sum())


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month, categorical=["PULocationID", "DOLocationID"])
