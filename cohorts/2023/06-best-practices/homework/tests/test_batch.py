from datetime import datetime
import pandas as pd
import pickle
import pytest


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


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


data = [
    (None, None, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2), dt(1, 10)),
    (1, 2, dt(2, 2), dt(2, 3)),
    (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
    (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
]

columns = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
]


df = pd.DataFrame(data, columns=columns)


class TestPrepareData(object):
    @pytest.mark.parametrize("df", [(df)])
    def test_with_null_data(self, df):
        """Test with pre-defined data."""
        expected_columns = ["ride_id", "predicted_duration"]
        expected_data = [
            ("2022/02_0", 24.781802),
            ("2022/02_1", 0.617543),
            ("2022/02_2", 6.108105),
        ]
        expected_df = pd.DataFrame(expected_data, columns=expected_columns)

        # Manually add the read_data() function transformations
        df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
        df["duration"] = df.duration.dt.total_seconds() / 60

        categorical = ["PULocationID", "DOLocationID"]

        df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

        df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

        # Execute the function prepare_data()
        actual_df = prepare_data(df, 2022, 2, ["PULocationID", "DOLocationID"])

        print(actual_df)
        pd.testing.assert_frame_equal(actual_df, expected_df)
