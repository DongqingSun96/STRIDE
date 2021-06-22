# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-06-22 08:42:54
# @Last Modified by:   Dongqing Sun
# @Last Modified time: 2021-06-22 19:20:12


import os
import matplotlib
import numpy as np
import pandas as pd

from scipy.spatial import distance
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DefaulfColorPalette = [
    "#E58606", "#5D69B1", "#52BCA3", "#99C945", "#CC61B0", "#24796C",
    "#DAA51B", "#2F8AC4", "#764E9F", "#ED645A", "#A5AA99", "#BCBD22",
    "#B279A2", "#EECA3B", "#17BECF", "#FF9DA6", "#778AAE", "#1B9E77",
    "#A6761D", "#526A83", "#B82E2E", "#80B1D3", "#68855C", "#D95F02",
    "#BEBADA", "#AF6458", "#D9AF6B", "#9C9C5E", "#625377", "#8C785D",
    "#88CCEE", "#E73F74", "#FFFFB3", "#CCEBC5", "#332288", "#A65628"
]


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
    group_input.add_argument("--plot", dest = "plot", action = "store_true",
        help = "Whether or not to visualize the neighbourhood analysis result. "
        "If set, the neighbourhood analysis result will be visualized. ")
    group_input.add_argument("--pt-size", dest = "pt_size", default = 2, type = float,
        help = "The size of point in the plot (only needed when 'plot' is set True). "
        "The size is set as 2 by default. "
        "For recommendation, the size should be set from 1 to 4. ")

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
    dist_min = np.quantile(dist_mat, q = st_loc_df.shape[0]/len(dist_mat))
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


def ClusterScatterPlot(st_loc_df, out_dir, out_prefix, pt_size = 2):
    '''
    Draw scatter pie plot to visualize the deconvolution result
    '''
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
    plot_file = os.path.join(out_dir, "%s_cluster_scatter_plot.pdf" %(out_prefix))
    fig.savefig(plot_file, bbox_inches = "tight")
    plt.close(fig)


def Clustering(deconv_res_file, st_loc_file, out_dir, out_prefix, plot, weight = 0.5, n_clusters = 5, pt_size = 2):
    st_deconv_df = pd.read_csv(deconv_res_file, sep = "\t", index_col = 0, header = 0)
    st_loc_df = pd.read_csv(st_loc_file, sep = "\t", index_col = 0, header = 0)
    neighbour_dict = FindNeighbours(st_loc_df)
    neighbour_frac_df = NeighbourAvgFrac(st_deconv_df, neighbour_dict)
    combined_frac_df = pd.concat([weight*st_deconv_df, (1-weight)*neighbour_frac_df], axis=1)
    kmeans_res = KMeans(n_clusters = n_clusters, random_state = 0, max_iter = 1000).fit(combined_frac_df)
    st_loc_df["Cluster"] = kmeans_res.labels_
    st_cluster_df = st_loc_df["Cluster"]
    cluster_file = os.path.join(out_dir, "%s_spot_cluster.txt" %(out_prefix))
    st_cluster_df.to_csv(cluster_file, sep = "\t")
    if plot:
        ClusterScatterPlot(st_loc_df, out_dir, out_prefix, pt_size = pt_size)

