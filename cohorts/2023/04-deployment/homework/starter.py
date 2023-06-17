#!/usr/bin/env python
# coding: utf-8

# Install packages needed
import pickle
import pandas as pd
import os, sys


def main(*args):
    # Load the model present in the homework directory
    with open("model.bin", "rb") as f_in:
        dv, model = pickle.load(f_in)

    categorical = ["PULocationID", "DOLocationID"]

    def read_data(filename):
        """Read data given a pasth"""
        df = pd.read_parquet(filename)

        df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df["duration"] = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

        return df

    year = args[0]
    month = args[1]

    # Read the yellow taxi data from March 2022
    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet"
    )
    # Transform using the dictionary vectorizer
    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)

    # Get the predictions of the model and print the standard deviation
    y_pred = model.predict(X_val)

    # df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    # # Create the df_result that contains two columns; "ride_id" and "pred"
    # df_result = pd.concat(
    #     [
    #         pd.Series(df["ride_id"]).reset_index(drop=True),
    #         pd.Series(y_pred).reset_index(drop=True),
    #     ],
    #     axis=1,
    # ).rename({0: " pred"}, axis=1)

    # # Define the name of the output file
    # output_file = os.getcwd() + "/output_file.parquet"

    # # Save the df_result as a parquet file
    # df_result.to_parquet(output_file, engine="pyarrow", compression=None, index=False)

    # # Get the size of the parquet file
    # os.path.getsize("./output_file.parquet")

    print("Mean prediction", y_pred.mean())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
