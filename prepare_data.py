import pandas as pd

def prepare_data(dataset):
    df = pd.read_csv(dataset)
    df = df.drop(df.columns[[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]], axis=1)
    ts = df.groupby("Date")["MaxTemp"].sum().rename("temperature");
    return ts