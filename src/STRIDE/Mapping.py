# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-08-31 18:53:18
# @Last Modified by:   Dongqing Sun
# @Last Modified time: 2021-09-01 13:21:32


import os
import pandas as pd
import numpy as np

from scipy.spatial import distance_matrix


def MappingParser(subparsers):
    workflow = subparsers.add_parser("map", 
        help = "Identify similarest cells for spatial spots. ")

    group_input = workflow.add_argument_group("Input arguments")
    group_input.add_argument("--spot-topic-file", dest = "topic_st_file", required = True,
        help = "Location of the file which stores the topic distribution of spatial spots (i.e., outdir/outprefix_topic_spot_mat_ntopic.txt). ")
    group_input.add_argument("--model-dir", dest = "model_dir", required = True,
        help = "The path of 'model' directory generated by 'deconvolve' (i.e., outdir/model). ")
    group_input.add_argument("--ntop", dest = "ntop", default = 10, type = int,
        help = "The size of point in the plot (only needed when 'plot' is set True). "
        "The size is set as 4 by default. "
        "For recommendation, the size should be increased as the number of point decreases. ")    


    group_output = workflow.add_argument_group("Output arguments")
    group_output.add_argument("--outdir", dest = "out_dir", default = ".", 
        help = "Path to the directory where the result file shall be stored. DEFAULT: current directory. ")
    group_output.add_argument("--outprefix", dest = "out_prefix", default = "STRIDE", 
        help = "Prefix of output files. DEFAULT: STRIDE. ")


def Mapping(topic_st_file, model_dir, ntop, out_dir, out_prefix):
    ntopic = topic_st_file.split("_topic_spot_mat_")[1].split(".")[0]
    topic_sc_file = os.path.join(model_dir, "topic_cell_mat_%s.txt" %ntopic)
    topic_sc_df = pd.read_csv(topic_sc_file, sep = "\t", index_col = 0)
    topic_st_df = pd.read_csv(topic_st_file, sep = "\t", index_col = 0)

    dist_mat = distance_matrix(topic_st_df.T, topic_sc_df.T)

    dist_mat_argsort = np.argsort(dist_mat, axis = 1)
    dist_mat_argsort_top = dist_mat_argsort[:,0:ntop]

    dist_df_argsort_top = pd.DataFrame(dist_mat_argsort_top, index = topic_st_df.columns)
    spot_cell_df = dist_df_argsort_top.apply(lambda x: topic_sc_df.columns[x])
    spot_cell_file = os.path.join(out_dir, "%s_spot_mapping_similar_%s_cell.txt" %(out_prefix, ntop))
    spot_cell_df.to_csv(spot_cell_file, sep = "\t", index = True, header = False)

