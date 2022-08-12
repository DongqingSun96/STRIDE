# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-06-22 19:54:39
# @Last Modified by:   dongqing
# @Last Modified time: 2022-03-22 14:52:59


import os
import pandas as pd
import numpy as np
import scipy.sparse
from scipy.spatial import distance_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch, Rectangle
import matplotlib.patches as mpatches
from ot.gromov import fused_gromov_wasserstein
from STRIDE.Plot import PieMarker, DefaulfColorPalette


def IntegrateParser(subparsers):
    workflow = subparsers.add_parser("integrate", 
        help = "Integrate multiple samples from the same tissue. ")

    group_input = workflow.add_argument_group("Input arguments")
    group_input.add_argument("--deconv-file", dest = "deconv_res_file_list", default = [], nargs = "+", type = str, required = True,
        help = "Location of the deconvolution result file of samples to be integrated. "
        "If the count matrices for individual samples are merged together to run 'STRIDE deconvolve', "
        "the deconvolution result can be directly provided here (i.e., outdir/outprefix_spot_celltype_frac.txt). "
        "If users perform deconvolution for each sample separately, please provide the result file list. "
        "For example, --deconv-file outdir/S1_spot_celltype_frac.txt outdir/S2_spot_celltype_frac.txt outdir/S3_spot_celltype_frac.txt. "
        "STRIDE will integrate the samples in the order of 'S1, S2, S3'. ")
    group_input.add_argument("--sample-id", dest = "sample_id_list", default = [], nargs = "+", type = str, 
        help = "Sample IDs to be integrated. "
        "If the count matrices for individual samples are merged together to run 'STRIDE deconvolve', "
        "please specify the samples to be integrated. For example, --sample-id 1 2 3. "
        "Otherwise, all samples will be integrated together. "
        "If users perform deconvolution for each sample separately, please ignore the argument. ")
    group_input.add_argument("--topic-file", dest = "topic_spot_file_list", default = [], nargs = "+", type = str, required = True,
        help = "Location of the topic distribution file for all samples to be integrated. "
        "If the count matrices for individual samples are merged together to run 'STRIDE deconvolve', "
        "the topic distribution file can be directly provided here (i.e., outdir/outprefix_topic_spot_mat_topicnumber.txt). "
        "If users perform deconvolution for each sample separately, please provide the topic distribution file list in the order of deconvolution files. "
        "For example, --topic-file outdir/S1_spot_topic_spot_mat_topicnumber.txt outdir/S2_topic_spot_mat_topicnumber.txt outdir/S3_topic_spot_mat_topicnumber.txt. ")
    group_input.add_argument("--st-loc", dest = "st_loc_file_list", default = [], nargs = "+", type = str, required = True,
        help = "Location of the ST spot information file. The file should be a tab-separated plain-text file with header. "
        "Users can provide either the merged location file or location file list of multiple samples, which depends on the way to perform deconvolution in the previous step. "
        "If users provide a merged location file, the file should have four columns. "
        "The first column should be the spot name, the second and the third column should be the row and column coordinate, "
        "and the last column should be the sample id. "
        "STRIDE will integrate all the samples by the alphabetic order of sample ids if the sample ids are of type 'strings'. "
        "If the sample ids provided are integers, STRIDE will integrate all the samples according to the natural order. "
        "If users proveide location file list, the files should be listed according to the order of provided deconvolution files. "
        "For example, --st-loc S1_location.txt S2_location.txt S3_location.txt ")
    group_input.add_argument("--plot", dest = "plot", action = "store_true",
        help = "Whether or not to visualize the integration analysis result. "
        "If set, the integration analysis result will be visualized. ")
    group_input.add_argument("--pt-size", dest = "pt_size", default = 2, type = float,
        help = "The size of point in the plot. "
        "The size is recommended to set from 2 to 4. "
        "DEFAULT: 5. ")

    group_analysis = workflow.add_argument_group("Analysis arguments")
    group_analysis.add_argument("--alpha", dest = "alpha", type = float, default = 0.5, 
        help = "The parameter to balance between the topic distribution similarity and spatial distance similarity of mapped spots. "
        "If 'alpha' is 1, only spatial distance similarity is taken into consideration. "
        "If 'alpha' is 0, only topic distribution similarity is taken into consideration. "
        "DEFAULT: 0.2. ")

    group_output = workflow.add_argument_group("Output arguments")
    group_output.add_argument("--outdir", dest = "out_dir", default = ".", 
        help = "Path to the directory where the result file shall be stored. DEFAULT: current directory. ")
    group_output.add_argument("--outprefix", dest = "out_prefix", default = "STRIDE", 
        help = "Prefix of output files. DEFAULT: STRIDE. ")


def KLDivergency(sample1_topic, sample2_topic):
    sample1_topic = sample1_topic/sample1_topic.sum(axis=1, keepdims=True)
    sample2_topic = sample2_topic/sample2_topic.sum(axis=1, keepdims=True)
    sample1_log = np.log(sample1_topic)
    sample2_log = np.log(sample2_topic)
    sample1_log_sample1 = np.array([np.apply_along_axis(lambda x: np.dot(x, np.log(x).T), 1, sample1_topic)])
    sample1_log_sample2 = np.dot(sample1_topic, sample2_log.T)
    kl_divergency = sample1_log_sample1.T - sample1_log_sample2
    return(kl_divergency)


def Alignment(sample1_id, sample2_id, st_loc_df, topic_spot_df, alpha = 0.2):
    # extract location of given slides
    s1_loc_df = st_loc_df[st_loc_df.iloc[:,2] == sample1_id]
    s2_loc_df = st_loc_df[st_loc_df.iloc[:,2] == sample2_id]
    s1_distance = distance_matrix(s1_loc_df.iloc[:,0:2], s1_loc_df.iloc[:,0:2])
    s2_distance = distance_matrix(s2_loc_df.iloc[:,0:2], s2_loc_df.iloc[:,0:2])
    # extract topic distribution of given slides
    s1_topic = np.array(topic_spot_df.T.loc[s1_loc_df.index,:]) + 0.0000000001
    s2_topic = np.array(topic_spot_df.T.loc[s2_loc_df.index,:]) + 0.0000000001
    # calculate distance of two topic matrices
    M = KLDivergency(s1_topic, s2_topic)
    # weight of spots
    p = np.ones((s1_loc_df.shape[0],))/s1_loc_df.shape[0]
    q = np.ones((s2_loc_df.shape[0],))/s2_loc_df.shape[0]
    # Computes the FGW transport between two slides
    pi = fused_gromov_wasserstein(M, s1_distance, s2_distance, p, q, loss_fun='square_loss', alpha= alpha)
    pi_df = pd.DataFrame(pi, index = s1_loc_df.index, columns = s2_loc_df.index)
    return(s1_loc_df, s2_loc_df, p, q, pi_df)


def IntegrationPlot(sample_all_loc_centered_df, st_deconv_df, sample_list, pt_size, out_dir, out_prefix):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if len(st_deconv_df.columns) > 15:
        color_pal = sns.color_palette("Spectral", len(st_deconv_df.columns))
    else:
        color_pal = DefaulfColorPalette
    for i in sample_all_loc_centered_df.index:
        loc_list = sample_all_loc_centered_df.loc[i,:].tolist()
        frac_list = st_deconv_df.loc[i,:]
        point_marker_list = PieMarker(loc_list, frac_list, pt_size**2, color_pal)
        for point_marker in point_marker_list:
            ax.scatter(point_marker[0], point_marker[1], point_marker[2], **point_marker[3])
    # adjust axes
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])
    plt.xticks(list(range(len(sample_list))), sample_list)
    ax.invert_xaxis()
    ax.set_xlabel(sample_all_loc_centered_df.columns[0])
    ax.grid(False)
    # add legends
    celltypes = st_deconv_df.columns
    patch_list = []
    for i in range(len(celltypes)):
        patch_list.append(mpatches.Patch(facecolor = color_pal[i], label=celltypes[i], edgecolor = "darkgrey", linewidth=0.1))
    ax.legend(handles = patch_list, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize = 'small', frameon = False,
        handlelength=1, handleheight=1)
    # save figure
    fig_file = os.path.join(out_dir, "%s_integration_plot.pdf" %(out_prefix))
    fig.savefig(fig_file, bbox_inches = "tight")


def Integrate(topic_spot_file_list, sample_id_list, deconv_res_file_list, st_loc_file_list, alpha, plot, pt_size, out_dir, out_prefix):
    if len(topic_spot_file_list) == 1:
        topic_spot_file = topic_spot_file_list[0]
        deconv_res_file = deconv_res_file_list[0]
        st_loc_file = st_loc_file_list[0]
        topic_spot_df = pd.read_csv(topic_spot_file, sep = "\t", index_col = 0, header = 0)
        st_deconv_df = pd.read_csv(deconv_res_file, sep = "\t", index_col = 0, header = 0)
        st_loc_df = pd.read_csv(st_loc_file, sep = '\t', index_col = 0, header = 0)
        if sample_id_list:
            if st_loc_df.iloc[:,2].dtype == np.int64:
                sample_id_list = [int(sample) for sample in sample_id_list]
            st_loc_df = st_loc_df.loc[st_loc_df.iloc[:,2].isin(sample_id_list), :]
            st_deconv_df = st_deconv_df.loc[st_loc_df.index, :]
            topic_spot_df = topic_spot_df.loc[:, st_loc_df.index]
            sample_list = sample_id_list
        else:
            sample_list = sorted(list(set(st_loc_df.iloc[:,2])))
    else:
        topic_spot_df_list = []
        st_deconv_df_list = []
        st_loc_df_list = []
        for s in range(0, len(topic_spot_file_list)):
            sample_id = "S%s" %(s+1)
            topic_spot_file = topic_spot_file_list[s]
            deconv_res_file = deconv_res_file_list[s]
            st_loc_file = st_loc_file_list[s]
            topic_spot_df = pd.read_csv(topic_spot_file, sep = "\t", index_col = 0, header = 0)
            topic_spot_df = topic_spot_df.rename(columns = dict(zip(topic_spot_df.columns, ["%s_%s" %(sample_id, i) for i in topic_spot_df.columns])))
            st_deconv_df = pd.read_csv(deconv_res_file, sep = "\t", index_col = 0, header = 0)
            st_deconv_df = st_deconv_df.rename(index = dict(zip(st_deconv_df.index, ["%s_%s" %(sample_id, i) for i in st_deconv_df.index])))
            st_loc_df = pd.read_csv(st_loc_file, sep = '\t', index_col = 0, header = 0)
            st_loc_df = st_loc_df.rename(index = dict(zip(st_loc_df.index, ["%s_%s" %(sample_id, i) for i in st_loc_df.index])))
            st_loc_df["Sample"] = sample_id
            topic_spot_df_list.append(topic_spot_df)
            st_deconv_df_list.append(st_deconv_df)
            st_loc_df_list.append(st_loc_df)
        topic_spot_df = pd.concat(topic_spot_df_list, axis = 1)
        st_deconv_df = pd.concat(st_deconv_df_list)
        st_loc_df = pd.concat(st_loc_df_list)
        st_loc_df = st_loc_df.iloc[:,[0,1,2]]
        sample_list = sorted(list(set(st_loc_df.iloc[:,2])))
    sample_1st = st_loc_df[st_loc_df.iloc[:,2] == sample_list[0]]
    sample_1st_array = np.array(sample_1st.iloc[:, [0, 1]]).T
    g_1st = np.ones((sample_1st.shape[0], 1))/sample_1st.shape[0]
    sample_1st_centered = sample_1st_array - np.dot(sample_1st_array, g_1st)
    sample_1st_centered_df = sample_1st.copy()
    sample_1st_centered_df[["X_trans","Y_trans"]] = sample_1st_centered.T
    sample_all_loc_centered_df = sample_1st_centered_df
    R = np.identity(2)
    alignment_dir = os.path.join(out_dir, "Pairewise_alignment")
    if not os.path.exists(alignment_dir):
        os.makedirs(alignment_dir)
    for i in range(len(sample_list)-1):
        print("Computing transport between %s and %s" %(sample_list[i], sample_list[i + 1]))
        s1_loc_df, s2_loc_df, g1, g2, pi_df = Alignment(sample1_id = sample_list[i], sample2_id = sample_list[i + 1], 
            st_loc_df = st_loc_df, topic_spot_df = topic_spot_df, alpha = alpha)
        alignment_file = os.path.join(alignment_dir, "%s_%s.txt" %(sample_list[i], sample_list[i + 1]))
        pi_df.to_csv(alignment_file, sep = "\t")
        pi_array = np.array(pi_df)
        s1_array = np.array(s1_loc_df.iloc[:, [0,1]]).T
        s2_array = np.array(s2_loc_df.iloc[:, [0,1]]).T
        # Center two slides
        s1_centered = s1_array - np.dot(s1_array, np.array([g1.tolist()]).T)
        s2_centered = s2_array - np.dot(s2_array, np.array([g2.tolist()]).T)
        # compute optimized rotation matrix R using SVD
        Object_array = np.dot(np.dot(s2_centered, pi_array.T), s1_centered.T)
        u, s, vh = np.linalg.svd(Object_array, full_matrices=True)
        R_new = np.dot(u,vh.T)
        R = np.dot(R, R_new)
        # Get the rotated slide2
        s2_centered_rotated = np.dot(R, s2_centered)
        s2_loc_centered_df = s2_loc_df.copy()
        s2_loc_centered_df[["X_trans","Y_trans"]] = s2_centered_rotated.T
        sample_all_loc_centered_df = pd.concat([sample_all_loc_centered_df, s2_loc_centered_df], axis=0)
    loc_trans_file = os.path.join(out_dir, "%s_integrated_location.txt" %(out_prefix))
    sample_all_loc_centered_df = sample_all_loc_centered_df[[sample_all_loc_centered_df.columns[2],"X_trans","Y_trans"]]
    sample_all_loc_centered_df.to_csv(loc_trans_file, sep = "\t")
    if plot:
        sns.set_context("notebook", font_scale = 1.2)
        for i, sample_id in enumerate(sample_list):
            sample_all_loc_centered_df.iloc[sample_all_loc_centered_df.iloc[:,0] == sample_id, 0] = i
        sample_all_loc_centered_df.iloc[:,0] = sample_all_loc_centered_df.iloc[:,0].astype(str).astype(int)
        IntegrationPlot(sample_all_loc_centered_df, st_deconv_df, sample_list, pt_size, out_dir, out_prefix)

