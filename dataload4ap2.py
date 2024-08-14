import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_load4ap1 import appendix1
from sklearn.preprocessing import StandardScaler
import os


class appendix2:
    def __init__(self):
        self.pathr = "datas/ap2_r_all.csv"
        self.pathw = "datas/ap2_w_all.csv"
        self.ap1 = appendix1()


class grapes(appendix2):
    def __init__(self):
        super().__init__()
        self.appearence = [
            f"g_{col}" for col in ["gsm", "bl", "gpm", "gp_l", "gp_a", "gp_b", "hs"]
        ] + ["name"]
        self.flavor = [f"g_{col}" for col in ["gs", "zf", "dn", "fx", "hs"]] + ["name"]
        self.process = [f"g_{col}" for col in ["zt", "hyt", "zs", "ph", "dn", "hb"]] + [
            "name"
        ]
        self.health = [f"g_{col}" for col in ["pt", "bl", "ht", "aj", "db", "vc"]] + [
            "name"
        ]

    def average_r(self):
        aver_df, _, _, _ = self.ap1.sum_group()
        average = aver_df.mean()
        return average

    def appearence_r(self):
        df = pd.read_csv(self.pathr, index_col=None, usecols=self.appearence)
        return df

    def appearence_w(self):
        df = pd.read_csv(self.pathw, index_col=None, usecols=self.appearence)
        return df

    def flavor_r(self):
        df = pd.read_csv(self.pathr, index_col=None, usecols=self.flavor)
        return df

    def flavor_w(self):
        df = pd.read_csv(self.pathw, index_col=None, usecols=self.flavor)
        return df

    def process_r(self):
        df = pd.read_csv(self.pathr, index_col=None, usecols=self.process)
        return df

    def process_w(self):
        df = pd.read_csv(self.pathw, index_col=None, usecols=self.process)
        return df

    def health_r(self):
        df = pd.read_csv(self.pathr, index_col=None, usecols=self.health)
        return df

    def health_w(self):
        df = pd.read_csv(self.pathw, index_col=None, usecols=self.health)
        return df

    def min_data_standardization(self, df, col):
        min_val = df[col].min()
        df[col] = df[col].apply(lambda x: x - min_val)
        return None

    def interval_data_standardization(self, df, col, a, b):
        min_val = df[col].min()
        max_val = df[col].max()
        c = max(a - min_val, max_val - b)

        def f(x):
            if x < a:
                return 1 - ((a - x) / c)
            elif x > b:
                return 1 - ((x - b) / c)
            else:
                return 1

        df[col] = df[col].apply(f)
        return None

    def median_data_standardization(self, df, col):
        max_val = df[col].max()
        min_val = df[col].min()
        median = (max_val + min_val) / 2

        def f(x):
            if x < median:
                return 2 * ((x - min_val) / (max_val - min_val))
            else:
                return 2 * ((max_val - x) / (max_val - min_val))

        df[col] = df[col].apply(f)
        return None

    def g_data_standardization(self, df):
        columns = df.columns
        min_list = "hb"
        interval_list = ("ph", "zs", "gs")
        median_list = "dn"
        for col in columns:
            if col in min_list:
                self.min_data_standardization(df, col)
            elif col in interval_list:
                if col == "ph":
                    self.interval_data_standardization(df, col, 6, 8)
                elif col == "zs":
                    self.interval_data_standardization(df, col, 8, 12)
                else:
                    self.interval_data_standardization(df, col, 25, 30)
            elif col in median_list:
                self.median_data_standardization(df, col)

        std = StandardScaler()
        df[columns[1:]] = std.fit_transform(df[columns[1:]])
        return df

    def kmeans(self, df, k=4):
        os.environ["OMP_NUM_THREADS"] = "1"

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df.drop("name", axis=1))
        df["cluster"] = kmeans.labels_
        metrix = pd.DataFrame({"cluster": kmeans.labels_})
        for col in df.columns:
            if (col != "name") and (col != "cluster"):
                df[f"{col}_center"] = df.groupby("cluster")[col].transform("mean")
                metrix[f"{col}_center"] = df[f"{col}_center"]
        metrix = metrix.groupby(["cluster"]).mean().reset_index()
        metrix.drop("cluster", axis=1, inplace=True)
        return df, metrix

    def rank(self, df):
        mapping = {0: 1, 1: 4, 2: 7, 3: 10}
        for col in df.columns:
            df[col] = df[col].rank(method="first").astype(int) - 1
            df[col] = df[col].map(mapping)
            df["sum"] = df.sum(axis=1)
        metric = (df["sum"].rank(method="first").astype(int) - 1).map(mapping).tolist()
        df["sum"] = df.iloc[:, :-1].sum(axis=1)
        return df, metric


class wine(appendix2):

    def __init__(self):
        super().__init__()
        
