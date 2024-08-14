import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from data_load4ap1 import appendix1
from sklearn.preprocessing import StandardScaler
import os
from scipy.stats import pearsonr
import statsmodels.api as sm
from tqdm import tqdm
import warnings

class appendix2:
    def __init__(self):
        self.pathr = "datas/ap2_r_all.csv"
        self.pathw = "datas/ap2_w_all.csv"
        self.ap1 = appendix1()
    
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


class grapes(appendix2):
    def __init__(self):
        super().__init__()
        self.app = ["gsm", "bl", "gpm", "gp_l", "gp_a", "gp_b", "hs"]
        self.fla = ["gs", "zf", "dn", "fx", "hs"]
        self.pro = ["zt", "hyt", "zs", "ph", "dn", "hb"]
        self.hea = ["pt", "bl", "ht", "aj", "db", "vc"]


        self.appearence = [f"g_{col}" for col in self.app] + ["name"]
        self.flavor = [f"g_{col}" for col in self.fla] + ["name"]
        self.process = [f"g_{col}" for col in self.pro] + ["name"]
        self.health = [f"g_{col}" for col in self.hea] + ["name"]

    def classify(self):
        print("appearence:", self.app)
        print("flavor:", self.fla)
        print("process:", self.pro)
        print("health:", self.hea)
        print("all:")

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

    def g_data_standardization(self, df):
        columns = df.columns
        min_list = "g_hb"
        interval_list = ("g_ph", "g_zs", "g_gs")
        median_list = "g_dn"
        for col in columns:
            if col in min_list:
                self.min_data_standardization(df, col)
            elif col in interval_list:
                if col == "g_ph":
                    self.interval_data_standardization(df, col, 6, 8)
                elif col == "g_zs":
                    self.interval_data_standardization(df, col, 8, 12)
                else:
                    self.interval_data_standardization(df, col, 25, 30)
            elif col in median_list:
                self.median_data_standardization(df, col)

        std = StandardScaler()
        df[columns[1:]] = std.fit_transform(df[columns[1:]])
        return df

    def kmeans(self, df, k=4):
        warnings.filterwarnings("ignore")
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
        warnings.filterwarnings("default")
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
        self.names = ["hs","dn","zf","jz","bl","dp","sz_l","sz_a","sz_b"]
        self.columns=[f"w_{col}" for col in self.names] + ["name"]

    def read_r(self):
        df = pd.read_csv(self.pathr, index_col=None, usecols=self.columns)
        return df
    
    def w_data_standardization(self, df):
        columns = df.columns
        min_list = "w_hb"
        interval_list = ("w_ph", "w_zs", "w_gs")
        median_list = "w_dn"
        for col in columns:
            if col in min_list:
                self.min_data_standardization(df, col)
            elif col in interval_list:
                if col == "w_ph":
                    self.interval_data_standardization(df, col, 6, 8)
                elif col == "w_zs":
                    self.interval_data_standardization(df, col, 8, 12)
                else:
                    self.interval_data_standardization(df, col, 25, 30)
            elif col in median_list:
                self.median_data_standardization(df, col)

        std = StandardScaler()
        df[columns[1:]] = std.fit_transform(df[columns[1:]])
        return df
    

class part4q3(appendix2):
    def __init__(self):
        super().__init__()
    
    def read_all_r(self):
        df=pd.read_csv(self.pathr, index_col=None)
        columns=df.columns
        g,w=[],[]
        for col in columns:
            if col.startswith("g_"):
                g.append(col)
            elif col.startswith("w_"):
                w.append(col)
        #print("grapes:",g)
        #print("wine:",w)
        return df,g,w
    
    def pearson_cal(self,df,g,w):
        pearson_dfs={w_col:None for w_col in w}
        for w_col in w:
            corrs,p_values,is_passed90=[],[],[]
            corrs,p_values=[],[]
            for g_col in g:
                corr, p_value = pearsonr(df[g_col], df[w_col])
                corrs.append(corr)
                p_values.append(p_value)
                is_passed90.append(p_value<=0.1)
            w_df=pd.DataFrame({"corr":corrs,"p_value":p_values,"is_passed90":is_passed90},index=g)
            w_df.sort_values(by="p_value",ascending=True,inplace=True)
            w_df=w_df[w_df["is_passed90"]==True].T
            w_df.index.name = w_col
            pearson_dfs[w_col]=w_df
        return pearson_dfs
    
    def all_data_standardization(self,df):
        df=grapes().g_data_standardization(df)
        df=wine().w_data_standardization(df)
        return df
    
    def forward_selection(self,X, y, threshold_in=0.1):
        initial_features = []
        best_features = list(initial_features)
        
        while True:
            remaining_features = list(set(X.columns) - set(best_features))
            new_pval = pd.Series(index=remaining_features)
            
            for new_column in remaining_features:
                model = sm.OLS(y, sm.add_constant(X[best_features + [new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
            
            min_p_value = new_pval.min()
            if min_p_value < threshold_in:
                best_features.append(new_pval.idxmin())
            else:
                break
                
        final_model = sm.OLS(y, sm.add_constant(X[best_features])).fit()

        return final_model,final_model.summary()


    def sbs_ols(self,df,w,support):
        summeries={}
        for w_col in tqdm(w, desc="Fitting Models"):
            select=support[w_col].columns.tolist()[1:6]
            y=df[w_col]
            X=df[select]
            model,stepwise_model = self.forward_selection(X,y)
            if model.rsquared_adj>=0.7:
                summeries[w_col]=stepwise_model
        print("Done")
        return summeries
    
        


    



    

    


    



