import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


class part4q4:
    def __init__(self):
        self.pathr="datas/ap3_r_all.csv"

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
        scaler = StandardScaler()
        df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
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
