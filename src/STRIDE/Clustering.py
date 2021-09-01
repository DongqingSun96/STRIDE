# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-06-22 08:42:54
# @Last Modified by:   Dongqing Sun
# @Last Modified time: 2021-09-01 13:59:45


import os
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.spatial import distance
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from STRIDE.Plot import DefaulfColorPalette

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']


def ClusterParser(subparsers):
    workflow = subparsers.add_parser("cluster", 
        help = "Neighbourhood analysis based on cell-type composition and local cell population ")

    group_input = workflow.add_argument_group("Input arguments")
    group_input.add_argument("--deconv-file", dest = "deconv_res_file", required = True,
        help = "Location of the deconvolution result file (i.e., outdir/outprefix_spot_celltype_frac.txt). ")
    group_input.add_argument("--st-loc", dest = "st_loc_file", required = True,
        help = "Location of the ST spot position file. "
        "The file should be a tab-separated plain-text file with header. "
        "The first column should be the spot name, and the second and the third column should be the row and column coordinate. ")
    group_input.add_argument("--sample-id", dest = "sample_id", default = None, type = str, 
        help = "If the count matrices are merged together to perform deconvolution, please specify the sample id of interest. ")
    group_input.add_argument("--plot", dest = "plot", action = "store_true",
        help = "Whether or not to visualize the neighbourhood analysis result. "
        "If set, the neighbourhood analysis result will be visualized. ")
    group_input.add_argument("--pt-size", dest = "pt_size", default = 4, type = float,
        help = "The size of point in the plot (only needed when 'plot' is set True). "
        "The size is set as 4 by default. "
        "For recommendation, the size should be increased as the number of point decreases. ")

    group_analysis = workflow.add_argument_group("Analysis arguments")
    group_analysis.add_argument("--weight", dest = "weight", type = float, default = 0.5, 
        help = "The parameter to balance between spots' own cell-type composition and local cell population. "
        "If the 'weight' is 1, only spots' own cell-type composition is taken into consideration. "
        "If the 'weight' is 0, only surrounding cell population is taken into consideration. "
        "By default, equal weights are given to both. "
        "DEFAULT: 0.5. ")
    group_analysis.add_argument("--ncluster", dest = "n_clusters", type = int, default = 5, 
        help = "The number of clusters to form. DEFAULT: 5. ")

    group_output = workflow.add_argument_group("Output arguments")
    group_output.add_argument("--outdir", dest = "out_dir", default = ".", 
        help = "Path to the directory where the result file shall be stored. DEFAULT: current directory. ")
    group_output.add_argument("--outprefix", dest = "out_prefix", default = "STRIDE", 
        help = "Prefix of output files. DEFAULT: STRIDE. ")


def FindNeighbours(st_loc_df):
    st_loc_df = st_loc_df.iloc[:,0:2]
    dist_mat = distance.pdist(st_loc_df, "euclidean")
    dist_min = np.quantile(dist_mat, q = 2*st_loc_df.shape[0]/len(dist_mat)) + 1
    kd_tree = KDTree(st_loc_df)
    neighbour_list = kd_tree.query_ball_tree(kd_tree, r = dist_min)
    neighbour_dict = {}
    for i in range(len(neighbour_list)):
        neighbour_list[i].remove(i)
        neighbour_dict[st_loc_df.index[i]] = st_loc_df.index[neighbour_list[i]]
    return(neighbour_dict)


def NeighbourAvgFrac(st_deconv_df, neighbour_dict):
    st_df = pd.DataFrame({'Spot':st_deconv_df.index}, index = st_deconv_df.index)
    neighbour_frac_df = st_df.apply(lambda x: st_deconv_df.loc[neighbour_dict[x[0]], ].mean(0), axis=1)
    return(neighbour_frac_df)


def ClusterScatterPlot(st_loc_df, out_dir, out_prefix, n_clusters, pt_size = 4):
    '''
    Draw scatter pie plot to visualize the deconvolution result
    '''
    sns.set_style("ticks")
    sns.set_context("notebook", font_scale=1.5)
    fig = plt.figure()
    ax = fig.add_subplot()
    clusters = sorted(list(set(st_loc_df["Cluster"])))
    cluster_list = []
    for i, label in enumerate(clusters):
        #add data points 
        points = plt.scatter(x = st_loc_df.loc[st_loc_df.index[st_loc_df['Cluster']==label], st_loc_df.columns[0]], 
                    y = st_loc_df.loc[st_loc_df.index[st_loc_df['Cluster']==label], st_loc_df.columns[1]], 
                    color = DefaulfColorPalette[i], alpha=1, s = pt_size**2)
        cluster_list.append(points)
    lgd = plt.legend(cluster_list, clusters, bbox_to_anchor=(1, 0.5), 
               loc='center left', markerscale = 1, frameon = False, handlelength=1, handleheight=1, fontsize = 'small')
    ax.axis('equal')
    ax.set_xlabel(st_loc_df.columns[0])
    ax.set_ylabel(st_loc_df.columns[1])
    ax.set_title(out_prefix, pad = 15)
    plot_file = os.path.join(out_dir, "%s_%scluster_scatter_plot.pdf" %(out_prefix, n_clusters))
    fig.savefig(plot_file, bbox_inches = "tight")
    plt.close(fig)


def Clustering(deconv_res_file, st_loc_file, sample_id, out_dir, out_prefix, plot, weight = 0.5, n_clusters = 5, pt_size = 4):
    st_deconv_df = pd.read_csv(deconv_res_file, sep = "\t", index_col = 0, header = 0)
    st_loc_df = pd.read_csv(st_loc_file, sep = "\t", index_col = 0, header = 0)
    if sample_id:
        st_loc_df.iloc[:,2] = st_loc_df.iloc[:,2].astype(str)
        st_loc_df = st_loc_df[st_loc_df.iloc[:,2] == sample_id]
        st_deconv_df = st_deconv_df.loc[st_loc_df.index,:]
    neighbour_dict = FindNeighbours(st_loc_df)
    neighbour_frac_df = NeighbourAvgFrac(st_deconv_df, neighbour_dict)
    combined_frac_df = pd.concat([weight*st_deconv_df, (1-weight)*neighbour_frac_df], axis=1)
    kmeans_res = KMeans(n_clusters = n_clusters, random_state = 0, max_iter = 1000).fit(combined_frac_df)
    st_deconv_df["Cluster"] = kmeans_res.labels_
    st_loc_df["Cluster"] = st_deconv_df.loc[st_loc_df.index, "Cluster"]
    st_cluster_df = st_loc_df["Cluster"]
    cluster_file = os.path.join(out_dir, "%s_spot_%scluster.txt" %(out_prefix, n_clusters))
    st_cluster_df.to_csv(cluster_file, sep = "\t")
    if plot:
        sns.set_context("notebook", font_scale=1.5)
        ClusterScatterPlot(st_loc_df, out_dir, out_prefix, n_clusters, pt_size = pt_size)

