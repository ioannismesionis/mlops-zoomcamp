from datetime import datetime
import pandas as pd
import pickle


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

year = 2022
month = 2

df_input = pd.DataFrame(data, columns=columns)
input_file = f"taxi_type_test_data=yellow_year={year:04d}_month={month:02d}.parquet"


df_input.to_parquet(input_file, engine="pyarrow", compression=None, index=False)
