import pandas as pd
import numpy as np
import pickle
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 

from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
from idtxl.estimators_python import PythonKraskovCMI
from idtxl import idtxl_io as io
from idtxl.visualise_graph import plot_network, plot_selected_vars

def individual_results(target, results, idx2name) :
    """ 
    Create a dataframe with main inference results. 
    """
    #Retrieve results
    res = results.get_single_target(target, fdr=False)

    name_sources = []
    info_delay = []
    for idx, lag in res['selected_vars_sources'] : 
        name_sources.append(idx2name[idx])
        info_delay.append(lag)

    df = pd.DataFrame({ 'Source' : name_sources, 
                        'Info delay' : info_delay,
                        'Target' : [idx2name[target] for i in range(len(name_sources))],
                        'TE' : res['selected_sources_te'],
                        'P-value' : res['selected_sources_pval'],              
                    })
    return df

def individual_results_all(targets_idx, results, idx2name) :
    """ 
    Concatenate inference results for all targets.
    Args :
        targets_idx : list of column indexes for targets 
    """
    df_list = []
    for target in targets_idx :
        df_tmp = individual_results(target, results, idx2name)
        df_list.append(df_tmp)

    df_res = pd.concat(df_list)
    df_res.set_index(['Target', 'Source'], inplace=True)
    return df_res

def joint_results(results, idx2name) :
    """
    Create joint inference results for all sources of each target.
    """
    te = []
    pvals = []
    targets = []

    for target in range(4) :
        res = results.get_single_target(target, fdr=False)
        te.append(res['omnibus_te'])
        pvals.append(res['omnibus_pval'])
        targets.append(idx2name[target])
    
    df_joint = pd.DataFrame({'Target' :targets,
                            'Joint TE' : te,
                            'P-value joint' : pvals}),
    return df_joint

def infer_graph_bTE(max_lag_sources, min_lag_sources, tau_sources, df, targets_idx, name_file) :
    """ 
    Network inference. 
    Args : 
        max_lag_sources : maximum number of past values considered for each source process
        min_lag_sources : minimum number of past values considered for each source process
        tau_sources : value for sub-sampling the past of the sources
        df : dataframe of interest 
        targets_idx : list of column indexes for targets 
        name_file : filename to save data 
    """
    #Transform data to IDTxl data format 
    data = Data(df, dim_order='sp')
    #Define settings for inference
    settings = {'cmi_estimator': 'PythonKraskovCMI',
                'n_perm_omnibus': 200,
                'alpha_omnibus': 0.05,
                'permute_in_time': True, 
                'max_lag_sources': max_lag_sources,      #Look up to 24 hours before
                'min_lag_sources': min_lag_sources, 
                'tau_sources' : tau_sources}       #Previous hour

    network_analysis = BivariateTE()

    #Infer network 
    results = network_analysis.analyse_network(settings=settings,
                                            data=data,
                                            targets=targets_idx)

    #Save results
    save_dir = "data/clean/transfer_entropy/"

    if not os.path.exists(save_dir) :
        os.mkdir(save_dir)

    pickle.dump(results, open(save_dir + name_file + '.p', 'wb'))


def infer_graph_mTE(max_lag_sources, min_lag_sources, tau_sources, df, name_file) :
    """ 
    Network inference for multivariate transfer entropy.  
    Args : 
        max_lag_sources : maximum number of past values considered for each source process
        min_lag_sources : minimum number of past values considered for each source process
        tau_sources : value for sub-sampling the past of the sources
        df : dataframe of interest 
        name_file : filename to save data 
    """
    #Transform data to IDTxl data format 
    data = Data(df, dim_order='sp')
    #Define settings for inference
    settings = {'cmi_estimator': 'PythonKraskovCMI',
                'n_perm_omnibus': 200,
                'alpha_omnibus': 0.05,
                'permute_in_time': True, 
                'max_lag_sources': max_lag_sources,      #Look up to 24 hours before
                'min_lag_sources': min_lag_sources, 
                'tau_sources' : tau_sources}       #Previous hour

    network_analysis = MultivariateTE()

    #Infer network 
    results = network_analysis.analyse_network(settings=settings,
                                                data=data)

    #Save results
    save_dir = "data/clean/transfer_entropy/"

    if not os.path.exists(save_dir) :
        os.mkdir(save_dir)

    pickle.dump(results, open(save_dir + name_file + '.p', 'wb'))

def load_results(path) : 
    """ 
    Load results from a previously infered graph. 
    """
    results = pickle.load(open(path, 'rb'))

    return results

def information_delay(df_res, name_file) : 
    """ 
    Display heatmap with information delay for each pair of processes (source, target)
    """
    df_res = df_res.reset_index()
    df_res.drop_duplicates(subset=['Source', 'Target'], inplace=True)
    df_res = df_res[['Source', 'Target', 'Info delay']]

    test = df_res.pivot(index="Source", columns="Target", values="Info delay")

    fig, ax = plt.subplots(figsize=(5, 5))

    ax = sns.heatmap(test, annot=True, linewidth=.5, cbar=False)
    ax.xaxis.tick_top()
    ax.set(xlabel="", ylabel="")

    if not os.path.exists("figures/") :
        os.mkdir("figures/")

    plt.savefig("figures/" + name_file + ".png")

    plt.show()