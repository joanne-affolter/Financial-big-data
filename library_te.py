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
    """Retrieve main inference results (p-values, information delay, estimated transfer entropy) for each pair of processes (source, target) for the given target.

    :param target: Target process index.
    :type target: Int
    :param results: Object containing inference results.
    :type results: Results object from IDTxl.
    :param idx2name: Dictionary mapping column indexes to column names.
    :type idx2name: Dict
    :return: Dataframe with inference results.
    :rtype: DataFrame
    """
    #Retrieve results for the given target. 
    res = results.get_single_target(target, fdr=False)

    name_sources = []
    info_delay = []

    #Retrieve information delay and p-values for each source process.
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
    """Retrieve main inference results (p-values, information delay, estimated transfer entropy) for each pair of processes (source, target), where 
    the target index is in the given list.

    :param targets_idx: List of target process indexes.
    :type targets_idx: List
    :param results: Object containing inference results.
    :type results: Results object from IDTxl.
    :param idx2name: Dictionary mapping column indexes to column names.
    :type idx2name: Dict
    :return: Dataframe with inference results.
    :rtype: DataFrame
    """
    df_list = []

    #For each target, retrieve inference results.
    for target in targets_idx :
        df_tmp = individual_results(target, results, idx2name)
        df_list.append(df_tmp)

    #Concatenate results for each target.
    df_res = pd.concat(df_list)
    df_res.set_index(['Target', 'Source'], inplace=True)

    return df_res

def joint_results(results, idx2name) :
    """Retrieve main inference results (joint transfer entropy and p-values) between all aggregated sources of a given target, for each target.

    :param results: Object containing inference results.
    :type results: Results object from IDTxl.
    :param idx2name: Dictionary mapping column indexes to column names.
    :type idx2name: Dict
    :return: Dataframe with inference results.
    :rtype: DataFrame
    """
    te = []
    pvals = []
    targets = []

    #Retrieve joint results for each target.
    for target in range(4) :
        res = results.get_single_target(target, fdr=False)
        te.append(res['omnibus_te'])
        pvals.append(res['omnibus_pval'])
        targets.append(idx2name[target])
    
    #Create dataframe with results.
    df_joint = pd.DataFrame({'Target' :targets,
                            'Joint TE' : te,
                            'P-value joint' : pvals}),
    return df_joint

def infer_graph_bTE(max_lag_sources, min_lag_sources, tau_sources, df, targets_idx, name_file) :
    """Network inference using bivariate Transfer Entropy (bTE). 

    :param max_lag_sources: Maximum number of past values considered for each source process
    :type max_lag_sources: Int
    :param min_lag_sources: Minimum number of past values considered for each source process
    :type min_lag_sources: Int
    :param tau_sources: Value for sub-sampling the past of the sources
    :type tau_sources: Int
    :param df: Dataframe containing the data for sources and targets 
    :type df: DataFrame
    :param targets_idx: List of column indexes for targets
    :type targets_idx: List
    :param name_file: Filename to save data
    :type name_file: String
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
    """Network inference using multivariate Transfer Entropy (mTE).

    :param max_lag_sources: Maximum number of past values considered for each source process
    :type max_lag_sources: Int
    :param min_lag_sources: Minimum number of past values considered for each source process
    :type min_lag_sources: Int
    :param tau_sources: Value for sub-sampling the past of the sources
    :type tau_sources: Int
    :param df: Dataframe containing the data for sources and targets
    :type df: DataFrame
    :param name_file: Filename to save data
    :type name_file: String
    """
    #Transform data to IDTxl data format 
    data = Data(df, dim_order='sp')

    #Define settings for inference
    settings = {'cmi_estimator': 'PythonKraskovCMI',
                'n_perm_omnibus': 200,
                'alpha_omnibus': 0.05,
                'permute_in_time': True, 
                'max_lag_sources': max_lag_sources,      
                'min_lag_sources': min_lag_sources, 
                'tau_sources' : tau_sources}       

    #Infer network
    network_analysis = MultivariateTE()
    results = network_analysis.analyse_network(settings=settings,
                                                data=data)

    #Save results
    save_dir = "data/clean/transfer_entropy/"

    if not os.path.exists(save_dir) :
        os.mkdir(save_dir)

    pickle.dump(results, open(save_dir + name_file + '.p', 'wb'))

def load_results(path) : 
    """Load results from a previously inferred graph. 

    :param path: Path to the file containing the results
    :type path: String
    :return: Object containing inference results.
    :rtype: Results object from IDTxl.
    """
    results = pickle.load(open(path, 'rb'))
    return results

def plot_information_delay(df, name_file) : 
    """Display heatmap with information delay for each pair of processes (source, target)

    :param df: Dataframe with information delay for each pair of processes (source, target)
    :type df: DataFrame
    :param name_file: Filename to save figure
    :type name_file: String
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    ax = sns.heatmap(df, annot=True, linewidth=.5, cbar=False)
    ax.xaxis.tick_top()
    ax.set(xlabel="", ylabel="")

    if not os.path.exists("figures/") :
        os.mkdir("figures/")

    plt.savefig("figures/" + name_file + ".png")

    plt.show()

def information_delay_bte(df_res, name_file) : 
    """Display heatmap with information delay for each pair of processes (source, target) for bTE inferred graph. 

    :param df_res: Dataframe with information delay for each pair of processes (source, target)
    :type df_res: DataFrame
    :param name_file: Filename to save figure
    :type name_file: String
    """
    df_res = df_res.reset_index()
    df_res.drop_duplicates(subset=['Source', 'Target'], inplace=True)
    df_res = df_res[['Source', 'Target', 'Info delay']]

    df = df_res.pivot(index="Source", columns="Target", values="Info delay")

    plot_information_delay(df, name_file)

def information_delay_mte(name_file) : 
    """Display heatmap with information delay for each pair of processes (source, target) for mTE inferred graph.

    :param name_file: Filename to save figure
    :type name_file: String
    """
    #Open file containing information delay for each pair of processes (source, target)
    data = pd.read_csv("data/clean/transfer_entropy/adjacency_mte.csv", delimiter=";")
    data.rename(columns={"Unnamed: 0" : "Source"}, inplace=True)
    data.set_index("Source", inplace=True)

    #Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 8))

    ax = sns.heatmap(data, annot=True, linewidth=.5, cbar=False)
    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='left')

    ax.set(xlabel="Target", ylabel="Source")

    if not os.path.exists("figures/") :
        os.mkdir("figures/")

    plt.savefig("figures/" + name_file + ".png")

    plt.show()

