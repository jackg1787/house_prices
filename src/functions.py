import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.base import  BaseEstimator, TransformerMixin

columns_for_modelling = []


def badrate_summary_categories(df, col, bad_col):
    summary = df.groupby(col)[bad_col].agg(['sum', 'count'])
    summary['rate'] = 100*summary['sum']/summary['count']
    summary.rename(columns = {'sum': 'n_positive', 'count': 'n_records'}, inplace=True)
    return summary


def plot_distro_per_cat(df, dist_col, cat_col, plt_args = {}, figsize= (15,10)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    df.loc[:, [cat_col, dist_col]].pivot(columns=cat_col)[dist_col].plot(
            kind="hist", ax=ax, **plt_args, title = 'distribution of {} per category of {}'.format(dist_col, cat_col))
            


def calculate_woe_for_column(newX, var,target = 'died', eps= 1e-10, min_rate = 0.05):
    """Calculate the weight of evidence per bin for a variable, and also check what is the gradient of the woe"""
    WoEs = []
    IV = -1
    trend = -1
    intercept = -1
    maxdiff = -1
    woe_trend = -1

    counts = newX.groupby(by=var, observed=True)[
        target].value_counts().sort_index()


    counts = counts.unstack(level=-1).fillna(0)
    counts.columns = counts.columns.to_series().map({0: "Good",
                                                     1: "Bad"})
    total = (counts["Good"]+counts["Bad"]).sum()
    total_good = counts["Good"].sum()
    total_bad = counts["Bad"].sum()
    # rate of bads per bin
    counts["Bad rate"] = counts["Bad"]/(counts["Good"]+counts["Bad"])
    # rate of entries per bin
    counts["Good+Bad"] = (counts["Good"]+counts["Bad"])/total
    # rate of goods over total of goods per bin
    counts["Good"] = counts["Good"]/total_good
    # rate of bads over total of bads per bin
    counts["Bad"] = counts["Bad"]/total_bad
    counts["Good%-Bad%"] = 100*(counts["Good"]-counts["Bad"])

    counts = counts[(counts["Good+Bad"] > min_rate) & (
        counts["Bad"] > 0) & (counts["Good"] > 0)].copy()

    counts["WoE"] = np.log((counts["Good"]+eps)/(counts["Bad"]+eps))
    counts["IV_i"] = 100*(counts["Good"]-counts["Bad"])*counts["WoE"]
    IV = counts["IV_i"].sum()

    # what way does our variable 'pull'?
    Yvals = counts.sort_index().loc[:, "WoE"].values
    Xvals = np.arange(len(Yvals))

    linear_reg = LinearRegression()
    linear_reg.fit(Xvals.reshape(-1, 1), Yvals.reshape(-1, 1))
    coef = linear_reg.coef_
    WoEs = Yvals.tolist()
    values_fit = Xvals*coef[0][0] + linear_reg.intercept_[0]
    trend = coef[0][0]
    intercept = linear_reg.intercept_[0]
    maxval = max(values_fit)
    minval = min(values_fit)
    maxdiff = maxval - minval

    woe_trend = get_woe_trend(counts.copy())

    return counts, woe_trend

def get_woe_trend(df):
    if df.shape[0] < 2:
        return 0
    elif (df['WoE'] == df['WoE'].cummax()).all():
        return +1
    elif (df['WoE'] == df['WoE'].cummin()).all():
        return -1
    else:
        return 0
    
class CorrelationAnalysis:
    """ could also add cramers v for categorical data. """
    def __init__(self,df, iv_table):

        
        self._corr_tables = {}
        
        self._corr_tables['pearson'] = pearson_df = df.corr(method='pearson')
        self._corr_tables['kendall'] = pearson_df = df.corr(method='kendall')
        self._corr_tables['spearman'] = pearson_df = df.corr(method='spearman')
        self._iv_table= iv_table
       
    def _get_correlated(self, corr_type, thresh):
        correlated = []
        correlated_dict = {}
        df = self._corr_tables[corr_type]
        for i, row in df.iterrows():
            for c, val in row.items():
                if c == i :
                    pass
                else:
                    if val> thresh:
                        print('{} correlated with {}'.format(i,c))
                        if i in correlated_dict.keys():
                            correlated_dict[i].append(c)
                        #elif c in correlated_dict.keys() and i in correlated_dict[c]:
                        #    pass
                        else:
                            correlated_dict[i] = [c]
        return correlated_dict
    
    @staticmethod
    def _update_dict(base, new):
        final = base.copy()
        for k in new.keys():
            if k in base.keys():
                final[k] = list(set(base[k]+new[k]))
            else:
                final[k] = new[k]
        return final
        
    def _get_best_by_iv(self, iv_table, var1, corr_with):
        """find the best feature by comparing the var1 to those it is correlated with"""
        var1_iv = iv_table.loc[var1, 'IV']
        others = iv_table.loc[(iv_table.index.isin(corr_with)) & (iv_table['IV'] > var1_iv), :]
        if others.empty:
            return var1, corr_with
        else:
            return others.sort_values(by='IV', ascending=False).index[0], [i for i in corr_with+[var1] if i != others.sort_values(by='IV', ascending=False).index[0]]
                                                                  

    def analyse_correlations(self, pearson_thresh=0.8, spearman_thresh=0.8, kendall_thresh=0.8):
        """keep the best feature according to iv if found to be correlated with other feats"""
        self._thresholds = {}
        self._thresholds['pearson'] = pearson_thresh
        self._thresholds['spearman'] = spearman_thresh
        self._thresholds['kendall'] = kendall_thresh
        correlations = {}
        all_corr = {}
        for corr_type in ['spearman','kendall','pearson']:
            print('running for {}'.format(corr_type))
            corrs = self._get_correlated(corr_type,self._thresholds[corr_type] )
            #print(corrs)
            correlations[corr_type] = corrs
            all_corr= self._update_dict(all_corr, corrs)
        
        if len(all_corr.keys())>0:
            print(all_corr)
            prev_rejects = []
            final_list = []
            for var1, corr_with in all_corr.items():
                print('filtering for {}'.format(var1))
                filtered_corr_with = list(set(corr_with)-set(prev_rejects))
                if var1 in prev_rejects:
                    pass
                elif len(filtered_corr_with) == 0:
                    final_list.append(var1)
                else:
                    selected, rejects = self._get_best_by_iv(self._iv_table, var1, corr_with)
                    print('rejected {}'.format(rejects))
                    #if we have rejected one that we previously accepted- 
                    # we want to go back and maybe keep the rejects from back then (assuming they are not correlated)
                    if len(set(rejects).intersection(final_list)):
                        print(' we are trying to reject one that was previously accepted- do something here (does not occur for titanic data)')
                    
                    final_list.append(selected)
                    prev_rejects.extend(rejects)
                    
        final = list(set(self._iv_table.index.to_list()) - set(prev_rejects)) + list(set(final_list))   
        return final, prev_rejects
    
    
class Preprocess(BaseEstimator, TransformerMixin):
    """pipeline step - used in api"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_=  X.copy()
        X_ = preprocess_data(X_)
        return X_
        
        
        
        
def preprocess_data(data, columns_for_modelling = columns_for_modelling):
    """steps used in the preprocessing step in the pipeline- used in api"""
    return data
    
        