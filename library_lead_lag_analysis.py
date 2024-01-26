import numpy as np
import os
import itertools
import pandas as pd
from datetime import datetime,timedelta
from statsmodels.stats.multitest import multipletests
import networkx as nx
from matplotlib import rcParams
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib.patches import Polygon
from matplotlib import patches, path, pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from library_pyRMT import clipped

### Marsili Giada Clustering

### This function was as code for the class FIN-525 "Financial Big Data"
def expand_grid_unique(x, y, include_equals=False):
    x = list(set(x))
    y = list(set(y))

    def g(i):
        z = [val for val in y if val not in x[:i - include_equals]]
        if z:
            return [x[i - 1]] + z

    combinations = [g(i) for i in range(1, len(x) + 1)]
    return [combo for combo in combinations if combo]

### This function was as code for the class FIN-525 "Financial Big Data"
def max_likelihood(c, n):
    if n > 1:

        return np.log(n / c) + (n - 1) * np.log((n * n - n) / (n * n - c))
    else:
        return 0
### This function was as code for the class FIN-525 "Financial Big Data"
def max_likelihood_list(cs, ns):
    Lc = {}
    for x in cs.keys():
        if ns[x] > 1:
            Lc[x] = np.log(ns[x] / cs[x]) + (ns[x] - 1) * np.log((ns[x] * ns[x] - ns[x]) / (ns[x] * ns[x] - cs[x]))
        else:
            Lc[x] = 0
    return Lc
### This function was as code for the class FIN-525 "Financial Big Data"
def find_max_improving_pair(C, cs, ns, i_s):

    N = len(i_s)
    Lc_new = {}
    Lc_old = max_likelihood_list(cs, ns)
    names_cs = list(cs.keys())
    max_impr = -1e10
    pair_max_improv = []
    
    for i in names_cs[:-1]:
        names_cs_j = names_cs[names_cs.index(i) + 1:]
        for j in names_cs_j:
            ns_new = ns[i] + ns[j]
            i_s_new = i_s[i] + i_s[j]
          #  ipdb.set_trace()
            cs_new = np.sum(C[np.ix_(i_s_new, i_s_new)])
            max_likelihood_new = max_likelihood(cs_new, ns_new)
            improvement = max_likelihood_new - Lc_old[i] - Lc_old[j]
            
            if improvement > max_impr:
                max_impr = improvement
                pair_max_improv = [i, j]
                Lc_max_impr = max_likelihood_new
#    print(pair_max_improv)
#    print(max_impr)
#    print(Lc_max_impr)
#    print("*************")

    return {"pair": pair_max_improv, "Lc_new": Lc_max_impr, "Lc_old": [Lc_old[x] for x in pair_max_improv]}

### This function was as code for the class FIN-525 "Financial Big Data"
def aggregate_clusters(C, only_log_likelihood_improving_merges=False):
    N = C.shape[0]
    cs = {i: 1 for i in range(N)}
    s_i = {i: [i] for i in range(N)}
    ns = {i: 1 for i in range(N)}
    i_s = {i: [i] for i in range(N)}
    Lc = {i: 0 for i in range(N)}
    
    all_pairs = [(i, j) for i in range(1, N + 1) for j in range(1, N + 1)]

    clusters = []
    for i in range(1, N):  # hierarchical merging
        improvement = find_max_improving_pair(C, cs, ns, i_s)
        Lc_old = improvement['Lc_old']
        Lc_new = improvement['Lc_new']
        
        if Lc_new < sum(Lc_old):
            print(" HALF CLUSTER  Lc.new > max(Lc.old)")
            # browser()
            
        if Lc_new <= max(Lc_old):
            print("Lc.new <= max(Lc.old), exiting")
            break
            
        pair = improvement['pair']
        s_i = [pair[0] if x == pair[1] else x for x in s_i]
    
        cluster1 = pair[0]
        cluster2 = pair[1]
        i_s[cluster1].extend(i_s[cluster2])  # merge the elements of the two clusters
        del i_s[cluster2]  # removes reference to merged cluster2
    
        ns[cluster1] += ns[cluster2]
        del ns[cluster2]
    
        cs[cluster1] = np.sum(C[i_s[cluster1]][:, i_s[cluster1]])  # sums C over the elements of cluster1
        del cs[cluster2]
    
        cs_vec = list(cs.values())
        ns_vec = list(ns.values())
    
        clusters.append({
            'Lc': max_likelihood_list(cs, ns),
            'pair_merged': pair,
            's_i': s_i,
            'i_s': i_s,
            'cs': cs,
            'ns': ns
        })
    
    last_clusters = clusters[-1]

    return last_clusters


def plot_giada_marsili_clustering(RMT_corrected=True):
    """
    This function generates Fig. 20 and 21 of our report.
    Arugments:
    RMT_corrected (bool) : True if correlation matrix of networks is cleaned with RMT before clustering"""
    llrs = pd.read_csv("data/clean/lead_lag_graphs/all_graphs.csv",header=None).values


    if RMT_corrected:
        #pyRMT cleaned correlation matrix
        correlation_matrix = clipped(llrs) 
    else : 
        correlation_matrix = np.corrcoef(llrs)
    #Apply clustering from course code.
    clusters = aggregate_clusters(correlation_matrix)

    all_clusters = clusters['i_s']
    inverted_clusters = {day: cluster for cluster, days in all_clusters.items() for day in days}
    cluster_reps = list(set(inverted_clusters.values()))

    rows = 12
    columns = 31
    matrix = np.full((rows, columns), np.nan)


    # Assign cluster colors to the matrix
    start_date = datetime(2021, 1, 1)
    index_dict = {element: index for index, element in enumerate(cluster_reps)}

    for day in range(0, 365):
        current_date = start_date + timedelta(days=day)
        month, date = current_date.month - 1, current_date.day - 1
        matrix[month, date] = index_dict.get(inverted_clusters.get(day, np.nan), np.nan)  # Adjust day + 1 to start from 1

    # Create a figure with 7 columns
    _, ax = plt.subplots()

    # Create a mask for non-NaN values
    mask = np.isnan(matrix)

    # Create a heatmap using seaborn with black borders only in the outer region
    sns.heatmap(matrix, cmap=sns.color_palette("Spectral", 3), xticklabels=True, yticklabels=True, ax=ax, mask=mask, linewidths=1, linecolor='black')

    # Set y-axis ticks to month names
    month_names = [datetime(2021, i + 1, 1).strftime('%B') for i in range(12)]
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(month_names, rotation=0)

    # Set x-axis ticks and labels
    ax.set_xticks(np.arange(columns) + 0.5)
    ax.set_xticklabels(np.arange(1, columns + 1))

    # Set minor ticks for both x and y axes
    ax.set_xticks(np.arange(31), minor=True)
    ax.set_yticks(np.arange(12), minor=True)

    # Customize the colorbar
    vmap = {i: chr(65 + i) for i in range(len(all_clusters))}
    n = len(vmap)
    colorbar = ax.collections[0].colorbar

    # Calculate positions to place the labels evenly distributed across the colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / n + r * i / n for i in range(n)])
    colorbar.set_ticklabels(list(vmap.values()))

    # Customize the grid
    ax.grid(True, which="minor", color="w", linewidth=2)
    ax.tick_params(which="minor", left=False, bottom=False)

    # Show the plot
    plt.show()

    plt.savefig(f"figures/clustering.png")



### Validating Statistical Significance of Lead-Lag Ratios


def empirical_cdf(x, data, bins=100):
    """Generates an empirical cdf of the data, used to calculate p-values from a null-model."""
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))
    
    # Linear interpolation to find the CDF at x
    cdf_at_x = np.interp(x, bin_edges[1:], cdf)
    
    return cdf_at_x

def remove_outliers(data):
    """Clips the distribution in quantiles 1% and 99% to avoid artifacts in the LLR calculation."""
    q1 = data.quantile(0.01)
    q3 = data.quantile(0.99)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]



def llr_plot(df_result, date, threshold):
    """This function generates Fig. 19 of our report."""
    #Class to have a polygon with rounded corners in matplotlib. Code originally from scleronomic in https://stackoverflow.com/questions/19270673/matplotlib-radius-in-polygon-edges-is-it-possible
    class RoundedPolygon(patches.PathPatch):
        def __init__(self, xy, pad, **kwargs):
            p = path.Path(*self.__round(xy=xy, pad=pad))
            super().__init__(path=p, **kwargs)

        def __round(self, xy, pad):
            n = len(xy)

            for i in range(0, n):

                x0, x1, x2 = np.atleast_1d(xy[i - 1], xy[i], xy[(i + 1) % n])

                d01, d12 = x1 - x0, x2 - x1
                d01, d12 = d01 / np.linalg.norm(d01), d12 / np.linalg.norm(d12)

                x00 = x0 + pad * d01
                x01 = x1 - pad * d01
                x10 = x1 + pad * d12
                x11 = x2 - pad * d12

                if i == 0:
                    verts = [x00, x01, x1, x10]
                else:
                    verts += [x01, x1, x10]
            codes = [path.Path.MOVETO] + n*[path.Path.LINETO, path.Path.CURVE3, path.Path.CURVE3]

            return np.atleast_1d(verts, codes)


    test = {}
    row = df_result.loc[date]
    # Loop through the columns and extract nodeX, nodeY, and the corresponding llrs value, to plot directed edges
    for column in df_result.columns:
        if 'llr_' in column:
            if 'keyword' in column:
                if 'rev' in column:
                    _, _, nodeY, _, nodeX  = column.split('_')

                else:
                    _, nodeX, _, nodeY = column.split('_')
                llrs_value = row[column]
            else:
                if 'rev' in column:
                    _, _, nodeY, nodeX, _ = column.split('_')

                else:
                    _, nodeX, nodeY, _ = column.split('_')
                llrs_value = row[column]
            if nodeX not in test:
                test[nodeX] = {}
            test[nodeX][nodeY] = llrs_value

    G = nx.DiGraph((k, v, {'weight': weight}) for k, vs in test.items() for v, weight in vs.items())
    adjacency_matrix = nx.to_pandas_adjacency(G).values.flatten()

    # Graph nodes
    ADA_nodes = ['binanceADA']
    ADA_keyword_nodes = ['Cardano']
    BTC_nodes = ['binanceBTC', 'bitstampBTC']
    BTC_keyword_nodes = ['btc']
    ETH_nodes = ['binanceETH','bitstampETH']
    ETH_keyword_nodes = ['Ethereum']
    LTC_nodes = ['binanceLTC','bitstampLTC']
    LTC_keyword_nodes = ['Litecoin']
    trading_node = ['trading']
    musk_node = ['Musk']
    FED_node = ['FED']
    node_list = ADA_nodes + BTC_nodes + ETH_nodes + LTC_nodes  + musk_node + FED_node + ADA_keyword_nodes + BTC_keyword_nodes + ETH_keyword_nodes + LTC_keyword_nodes

    def filter_edges(graph,threshold, all_keywords=False):
        """This function filters out edges that are not statistically significant
        Arguments:
        graph (nx.Graph) : the graph that will have edges removed
        threshold (float) : The p-value that is the threshold for statistical significant edges
        all_keywords (bool) : True if edges coming from statistically not significant keywords should also be filtered. False if only statistically significant keywords should be kept."""
        filtered_edges = []
        for edge in graph.edges(data=True):
            source, target, data = edge
            if all_keywords or (source in node_list and target in node_list):
                reverse_edge_exists = graph.has_edge(target, source)
                if not reverse_edge_exists and data['weight'] > threshold:
                    filtered_edges.append((source, target, data))
                else:
                    reverse_edge_weight = graph.get_edge_data(target, source).get('weight', 0)
                    original_edge_weight = data.get('weight', 0)
                    if original_edge_weight > reverse_edge_weight and data['weight'] > threshold:
                        filtered_edges.append((source, target, data))
        return filtered_edges
    filtered_edges = sorted(filter_edges(G,threshold), key=lambda x: x[2]['weight'], reverse=True)
    rcParams['font.size']

    #Generate a new graph with the filtered edge list.
    G2 = nx.DiGraph()
    G2.add_edges_from(filtered_edges)

    pos = graphviz_layout(G2,prog='twopi')

    #Node positions in the plot.
    pos = {'binanceETH': (0,2), 
        'binanceLTC': (1,0), 
        'binanceADA': (-1,0),
        'binanceBTC': (0,-2),
        'bitstampETH': (2,2),
        'bitstampBTC': (2,-2), 
        'bitstampLTC': (2,0),
        'Cardano' : (2,3),
        'Ethereum' : (1,3),
        'btc' : (0,3),
        'Litecoin' : (-1,3),
        'Musk' : (-1,4),
        'trading' : (0,4),
        'FED' : (1,4)}

    #Create legend handles to label nodes correctly. 
    nodes_handle1 = nx.draw_networkx_nodes(G2, pos, nodelist=ADA_nodes,label='ADA coin', node_color='#33a02c',node_size=15**2)
    ADA_keyword_nodes = nx.draw_networkx_nodes(G2, pos, nodelist=ADA_keyword_nodes,label='Cardano keyword', node_shape = 's', node_color='#33a02c',node_size=15**2)
    nodes_handle2 = nx.draw_networkx_nodes(G2, pos, nodelist=BTC_nodes,label='BTC coin', node_color='#1f78b4', node_size=15**2)
    BTC_keyword_nodes = nx.draw_networkx_nodes(G2, pos, nodelist=BTC_keyword_nodes,label='BTC keyword', node_shape = 's', node_color='#1f78b4', node_size=15**2)
    nodes_handle3 = nx.draw_networkx_nodes(G2, pos, nodelist=ETH_nodes,label='ETH coin', node_color='#e31a1c', node_size=15**2)
    ETH_keyword_nodes = nx.draw_networkx_nodes(G2, pos, nodelist=ETH_keyword_nodes,label='Ethereum keyword', node_shape = 's', node_color='#e31a1c', node_size=15**2)
    nodes_handle4 = nx.draw_networkx_nodes(G2, pos, nodelist=LTC_nodes,label='LTC coin', node_color='#6a3d9a', node_size=15**2)
    LTC_keyword_nodes = nx.draw_networkx_nodes(G2, pos, nodelist=LTC_keyword_nodes,label='Litecoin keyword',  node_shape = 's', node_color='#6a3d9a', node_size=15**2)
    nodes_handle5 = nx.draw_networkx_nodes(G2, pos, nodelist=musk_node,label='Musk keyword', node_shape = 's', node_color='#525252', node_size=15**2)
    nodes_handle6 = nx.draw_networkx_nodes(G2, pos, nodelist=trading_node,label='trading keyword', node_shape = 's', node_color='#737373', node_size=15**2)
    nodes_handle7 = nx.draw_networkx_nodes(G2, pos, nodelist=FED_node,label='FED keyword', node_shape = 's', linewidths=1,node_color='#bdbdbd', node_size=15**2)
    weights = np.array(list(nx.get_edge_attributes(G,'weight').values()))

    #Position of nodes in the plot.
    binance_vertices = np.array([[-1.5, 2.4], [1.5, 2.4], [1.5, -2.4], [-1.5, -2.4]])
    bitstamp_vertices = np.array([[1.6, 2.4], [2.4, 2.4], [2.4, -2.4], [1.6, -2.4]])
    GT_vertices = np.array([[-1.5, 4.4], [2.4, 4.4], [2.4, 2.5], [-1.5, 2.5]])

    #Position of patches corresponding to exchanges and Google Trends in the plot.
    binance_patch = RoundedPolygon(xy=binance_vertices, pad= 0.3, facecolor='white', edgecolor='black', linestyle='-',  linewidth=2,joinstyle='round', alpha=1)
    bitstamp_patch = RoundedPolygon(xy=bitstamp_vertices, pad= 0.3, facecolor='white', edgecolor='black', linestyle='-',  linewidth=2,joinstyle='round', alpha=0.7)
    GT_patch = RoundedPolygon(xy=GT_vertices, pad= 0.3, facecolor='white', edgecolor='black', linestyle='-',  linewidth=2,joinstyle='round', alpha=0.4)
    
    #Plot configs.
    plt.axis("off")
    plt.legend(handles=[nodes_handle1,nodes_handle2,nodes_handle3,nodes_handle4, nodes_handle5, nodes_handle6, nodes_handle7, ADA_keyword_nodes, BTC_keyword_nodes,ETH_keyword_nodes,LTC_keyword_nodes], loc="lower left",bbox_to_anchor=(1.05,0))
    plt.gca().add_patch(binance_patch)
    plt.gca().add_patch(bitstamp_patch)
    plt.gca().add_patch(GT_patch)
    nx.draw_networkx_edges(G2,pos,width=weights,connectionstyle='arc3, rad = 0.1')
    plt.text(-1.7, 3.6, 'Google Trends', fontsize=14, color='black', ha='center', va='center',rotation=90,alpha=0.4)
    plt.text(-1.7, 1.8, 'Binance', fontsize=14, color='black', ha='center', va='center',rotation=90)
    plt.text(2.6, 1.6, 'Bistamp', fontsize=14, color='black', ha='center', va='center',rotation=270,alpha=0.7)

    plt.ylim([-2.5,4.5])
    plt.xlim([-4,2.5])
    plt.tight_layout()
    plt.savefig(f"figures/lead_lag_graph_{date}")
    plt.show()

    def degrees_to_dataframe_row(graph):
        """This function is used to calculate centrality metrics of the graph for Table 7 in the report."""
        data = {}
        
        for node in graph.nodes:
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            data[f"In-Degree_{node}"] = in_degree
            data[f"Out-Degree_{node}"] = out_degree
        
        return pd.DataFrame([data])

    return adjacency_matrix, filtered_edges, node_list, degrees_to_dataframe_row(G2)

def longest_running_link(array):
    """This function is used to calculate lead-lag metrics of the graph for Table 7 in the report."""

    max_len = 0
    current_len = 0
    index = 0

    for index in range(len(array)):
        if array[index] == 0:
            max_len = max(max_len,current_len)
            current_len = 0
        else:
            current_len += 1
        index += 1

    max_len = max( max_len, current_len)
    return max_len


def fetch_lead_lag_ratios():
    """This function fetches an intermediary result to avoid duplicate computations : we save lead_lag_ratios of all 365 daily networks as vectors in order to compute their correlation and perform clustering."""

    directory = "data/clean/lead_lag_ratios"
    empirical_distribution_null_model = pd.read_csv(os.path.join(directory,"null_model.csv"))
    files = os.listdir(directory)
    llrs_files = [file for file in files if file.endswith('.csv') and 'null_model' not in file]
    df_result = pd.read_csv(os.path.join(directory,llrs_files[0]), index_col='date')
    df_result.rename(lambda x : x + "_"+ os.path.splitext(llrs_files[0])[0],inplace=True,axis=1)
    for file in llrs_files[1:]:
        df_temp = pd.read_csv(os.path.join(directory,file), index_col='date')
        df_temp.rename(lambda x : x + "_"+ os.path.splitext(file)[0],inplace=True,axis=1)
        df_result = df_result.join(df_temp)

    llr_columns = [col for col in df_result.columns if col.startswith('llr')]
    llr_df = df_result[llr_columns]
    flattened_data = pd.Series(llr_df.values.flatten())
    filtered_llrs = remove_outliers(flattened_data)
    statistically_significant_llr = np.array([llr for llr in filtered_llrs if multipletests(1 - empirical_cdf(llr, empirical_distribution_null_model['LLR'].values))[0]])
    minimum_statistically_significant_llr = np.min(statistically_significant_llr)
    return llr_df, minimum_statistically_significant_llr 