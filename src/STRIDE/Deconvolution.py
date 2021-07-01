# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-06-09 08:56:08
# @Last Modified by:   Dongqing Sun
# @Last Modified time: 2021-06-30 14:56:12


import os
import scipy
import gensim

import numpy as np
import pandas as pd
import argparse as ap

from gensim.models import LdaModel
from sklearn.preprocessing import StandardScaler

from STRIDE.ModelTrain import scProcess, stProcess, scLDA
from STRIDE.utility.MarkerFind import MarkerFind


def DeconvolveParser(subparsers):
    workflow = subparsers.add_parser("deconvolve", 
        help = "Decompose celltype proportion for spatial transcriptomics.")

    group_input = workflow.add_argument_group("Input arguments")
    group_input.add_argument("--sc-count", dest = "sc_count_file", required = True,
        help = "Location of the single-cell count matrix file. "
        "It could be '.h5' or tab-separated plain-text file with genes as rows and cells as columns. ")
    group_input.add_argument("--sc-celltype", dest = "sc_anno_file", required = True,
        help = "Location of the single-cell celltype annotation file. "
        "The file should be a tab-separated plain-text file without header. "
        "The first column should be the cell name, and the second column should be the corresponding celltype labels. ")
    group_input.add_argument("--st-count", dest = "st_count_file", required = True,
        help = "Location of the spatial gene count file. "
        "It could be '.h5' or tab-separated plain-text file with genes as rows and spots as columns. ")
    group_input.add_argument("--gene-use", dest = "gene_use", default = None,
        help = "Location of the gene list file used to train the model. "
        "It can also be specified as 'All', but it will take a longer time. "
        "If not specified, STRIDE will find differential marker genes for each celltype, and use them to run the model. ")

    group_output = workflow.add_argument_group("Output arguments")
    group_output.add_argument("--outdir", dest = "out_dir", default = ".", 
        help = "Path to the directory where the result file shall be stored. DEFAULT: current directory. ")
    group_output.add_argument("--outprefix", dest = "out_prefix", default = "STRIDE", 
        help = "Prefix of output files. DEFAULT: STRIDE. ")

    group_model = workflow.add_argument_group("Model arguments")
    group_model.add_argument("--sc-scale-factor", dest = "sc_scale_factor", type = float, default = None,
        help = "The scale factor for cell-level normalization. For example, 10000. "
        "If not specified, STRIDE will set the 75%% quantile of nCount as default. ")
    group_model.add_argument("--st-scale-factor", dest = "st_scale_factor", type = float, default = None,
        help = "The scale factor for spot-level normalization. For example, 10000. "
        "If not specified, STRIDE will set the 75%% quantile of nCount for ST as default. ")
    group_model.add_argument("--normalize", dest = "normalize", action = "store_true", 
        help = "Whether or not to normalize the single-cell and the spatial count matrix. "
        "If set, the two matrices will be normalized by the SD for each gene. ")
    group_model.add_argument("--ntopics", dest = "ntopics_list", default = [], nargs = "+",
        help = "Number of topics to train and test the model. STRIDE will automatically select the optimal topic number. "
        "Multiple numbers should be separated by space. For example, --ntopics 6 7 8 9 10 . "
        "If not specified, STRIDE will run several models with different topic numbers, and select the optimal one. ")


def SpatialDeconvolve(st_count_mat, st_count_genes, st_count_spots, genes_shared, model_selected, ntopics_selected, normalize, out_dir, out_prefix):
    st_count_genes_array = np.array(st_count_genes)
    st_count_genes_sorter = np.argsort(st_count_genes_array)
    if normalize:
        st_count_mat = StandardScaler(with_mean=False).fit_transform(st_count_mat.transpose()).transpose()
    genes_shared_array = np.array(genes_shared)
    genes_shared_index = st_count_genes_sorter[np.searchsorted(st_count_genes_array, genes_shared_array, sorter = st_count_genes_sorter)]
    st_count_mat_use = st_count_mat[genes_shared_index,:]
    st_corpus = []
    for i in range(st_count_mat_use.shape[1]):
        st_genes_nonzero_index = np.nonzero(st_count_mat_use[:,i])[0]
        st_genes_nonzero_count = st_count_mat_use[st_genes_nonzero_index, i].toarray().flatten().tolist()
        st_corpus.append(list(zip(st_genes_nonzero_index, st_genes_nonzero_count)))
    topic_spot_file = os.path.join(out_dir, "%s_topic_spot_mat_%s.npz" %(out_prefix, ntopics_selected))
    if os.path.exists(topic_spot_file):
        topic_spot_mat = scipy.sparse.load_npz(topic_spot_file)
    else:
        model_dir = os.path.join(out_dir, "model")
        model_file = os.path.join(model_dir, "lda_model_%s" %(ntopics_selected))
        lda = LdaModel.load(model_file)
        topic_spot = lda.get_document_topics(st_corpus)
        topic_spot_mat = gensim.matutils.corpus2csc(topic_spot)
        scipy.sparse.save_npz(topic_spot_file, topic_spot_mat)
    if model_selected == "Raw":
        spot_celltype_array_norm_df = SpatialDeconvolveRaw(st_count_spots, ntopics_selected, topic_spot_mat, out_dir)
    if model_selected == "Norm":
        spot_celltype_array_norm_df = SpatialDeconvolveNorm(st_count_spots, ntopics_selected, topic_spot_mat, out_dir)
    if model_selected == "NormBySD":
        spot_celltype_array_norm_df = SpatialDeconvolveNormBySD(st_count_spots, ntopics_selected, topic_spot_mat, out_dir)
    if model_selected == "Bayes":
        spot_celltype_array_norm_df = SpatialDeconvolveBayes(st_count_spots, ntopics_selected, topic_spot_mat, out_dir)
    if model_selected == "BayesNorm":
        spot_celltype_array_norm_df = SpatialDeconvolveBayesNorm(st_count_spots, ntopics_selected, topic_spot_mat, out_dir)
    spot_celltype_array_norm_df_file = os.path.join(out_dir, "%s_spot_celltype_frac.txt" %(out_prefix))
    spot_celltype_array_norm_df.to_csv(spot_celltype_array_norm_df_file, sep = "\t")
    return(spot_celltype_array_norm_df)


def SpatialDeconvolveRaw(st_count_spots, ntopics, topic_spot_mat, out_dir):
    # read the celltype-topic file
    model_dir = os.path.join(out_dir, "model")
    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat_%s.txt" %(ntopics))
    topic_celltype_df = pd.read_csv(topic_celltype_file, sep="\t", index_col = 0)
    celltype_topic_df = topic_celltype_df.transpose()
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_df.iloc[:,0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_norm = np.divide(spot_celltype_array, spot_celltype_array.sum(axis = 1)[:,None])
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_df.index
    spot_celltype_array_norm_df.index = st_count_spots

    return(spot_celltype_array_norm_df)


def SpatialDeconvolveNorm(st_count_spots, ntopics, topic_spot_mat, out_dir):
    # read the celltype-topic file
    model_dir = os.path.join(out_dir, "model")
    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat_%s.txt" %(ntopics))
    topic_celltype_df = pd.read_csv(topic_celltype_file, sep="\t", index_col = 0)
    celltype_topic_df = topic_celltype_df.transpose()
    celltype_topic_norm_df = np.divide(celltype_topic_df, celltype_topic_df.sum(axis = 0)[None,:])
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_norm_df.iloc[:,0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_norm = np.divide(spot_celltype_array, spot_celltype_array.sum(axis = 1)[:,None])
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_norm_df.index = st_count_spots

    return(spot_celltype_array_norm_df)


def SpatialDeconvolveNormBySD(st_count_spots, ntopics, topic_spot_mat, out_dir, out_prefix):
    # read the celltype-topic file
    model_dir = os.path.join(out_dir, "model")
    topic_celltype_file = os.path.join(model_dir,"topic_celltype_mat_%s.txt" %(ntopics))
    topic_celltype_df = pd.read_csv(topic_celltype_file, sep="\t", index_col = 0)
    celltype_topic_df = topic_celltype_df.transpose()
    celltype_topic_norm_array = StandardScaler(with_mean=False).fit_transform(celltype_topic_df)
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_norm_array[:,0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_norm = np.divide(spot_celltype_array, spot_celltype_array.sum(axis = 1)[:,None])
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_df.index
    spot_celltype_array_norm_df.index = st_count_spots

    return(spot_celltype_array_norm_df)


def SpatialDeconvolveBayes(st_count_spots, ntopics, topic_spot_mat, out_dir, out_prefix):
    # read the celltype-topic file
    model_dir = os.path.join(out_dir, "model")
    celltype_topic_file = os.path.join(model_dir,"celltype_topic_mat_bayes_%s.txt" %(ntopics))
    celltype_topic_bayes_df = pd.read_csv(celltype_topic_file, sep="\t", index_col = 0)
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_bayes_df.iloc[:,0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_norm = np.divide(spot_celltype_array, spot_celltype_array.sum(axis = 1)[:,None])
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_bayes_df.index
    spot_celltype_array_norm_df.index = st_count_spots

    return(spot_celltype_array_norm_df)


def SpatialDeconvolveBayesNorm(st_count_spots, ntopics, topic_spot_mat, out_dir, out_prefix):
    # read the celltype-topic file
    model_dir = os.path.join(out_dir, "model")
    celltype_topic_file = os.path.join(model_dir,"celltype_topic_mat_bayes_%s.txt" %(ntopics))
    celltype_topic_bayes_df = pd.read_csv(celltype_topic_file, sep="\t", index_col = 0)
    celltype_topic_norm_df = np.divide(celltype_topic_bayes_df, celltype_topic_bayes_df.sum(axis = 0)[None,:])
    # deconvolution
    celltype_spot_array = np.dot(celltype_topic_norm_df.iloc[:,0:topic_spot_mat.shape[0]], topic_spot_mat.toarray())
    spot_celltype_array = celltype_spot_array.transpose()
    spot_celltype_array_norm = np.divide(spot_celltype_array, spot_celltype_array.sum(axis = 1)[:,None])
    spot_celltype_array_norm_df = pd.DataFrame(spot_celltype_array_norm)
    spot_celltype_array_norm_df.columns = celltype_topic_norm_df.index
    spot_celltype_array_norm_df.index = st_count_spots
    
    return(spot_celltype_array_norm_df)


def Deconvolve(sc_count_file, sc_anno_file, st_count_file, sc_scale_factor, st_scale_factor,
               out_dir, out_prefix, normalize, gene_use = None, ntopics_list = None):
    print("Reading single-cell count matrix...")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sc_info = scProcess(sc_count_file, sc_anno_file, out_dir, out_prefix, sc_scale_factor)
    sc_count_scale_mat = sc_info["scale_matrix"]
    sc_count_raw_mat = sc_info["raw_matrix"]
    sc_count_genes = sc_info["genes"]
    sc_count_cells = sc_info["cells"]
    cell_celltype_dict = sc_info["cell_celltype"]
    print("Reading spatial count matrix...")
    st_info = stProcess(st_count_file, st_scale_factor)
    st_count_mat = st_info["scale_matrix"]
    st_count_genes = st_info["genes"]
    st_count_spots = st_info["spots"]
    findmarker = False
    if gene_use:
        if os.path.exists(gene_use):
            gene_use = open(gene_use, "r").readlines()
            gene_use = [i.strip() for i in gene_use]
        elif gene_use == "All":
            gene_use = gene_use
        else:
            print("The gene file doesn't exist. Identifying markers...")
            findmarker = True
    else:
        print("Identifying markers...")
        findmarker = True
    if findmarker:
        gene_use = MarkerFind(sc_count_raw_mat, sc_count_genes, sc_count_cells, sc_anno_file, ntop = 200)
    celltypes = set(cell_celltype_dict.values())
    if not ntopics_list:
        ntopics_list = list(range(len(celltypes), 3*len(celltypes)+1))
    print("Training topic model...")
    lda_res = scLDA(sc_count_scale_mat, sc_count_genes, sc_count_cells, cell_celltype_dict,
                    st_count_mat, st_count_genes, st_count_spots,
                    normalize, gene_use, ntopics_list, out_dir)
    genes_shared = lda_res["genes_shared"]
    model_selected = lda_res["model_selected"]
    ntopics_selected = lda_res["ntopics_selected"]
    print("Deconvolving spatial transcriptomics...")
    spot_celltype_array_norm_df = SpatialDeconvolve(st_count_mat, st_count_genes, st_count_spots, genes_shared, model_selected, ntopics_selected, normalize, out_dir, out_prefix)

