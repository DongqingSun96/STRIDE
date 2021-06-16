# -*- coding: utf-8 -*-
# @Author: Dongqing Sun
# @E-mail: Dongqingsun96@gmail.com
# @Date:   2021-06-16 14:22:51
# @Last Modified by:   Dongqing Sun
# @Last Modified time: 2021-06-16 17:14:30


import numpy as np
import pandas as pd
import scanpy as sc
import anndata


def MarkerFind(sc_count_mat, sc_count_genes, sc_count_cells, sc_anno_file, ntop = 200):
    '''
    Find markers for each celltype
    '''
    adata = anndata.AnnData(sc_count_mat.transpose(), obs = dict(obs_names = sc_count_cells), var = dict(var_names = sc_count_genes))
    # preprocess
    sc.pp.filter_cells(adata, min_genes = 200)
    sc.pp.filter_genes(adata, min_cells = 10)
    sc.pp.calculate_qc_metrics(adata, inplace = True)
    # normalize data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes = 2000)
    # add celltype info
    meta_data = pd.read_csv(sc_anno_file, sep = "\t", header = None, index_col = 0)
    meta_data.columns = ["Celltype"]
    adata.obs["Celltype"] = meta_data.loc[adata.obs.index, "Celltype"]
    # find marker genes for each cell-type
    sc.tl.rank_genes_groups(adata, groupby = 'Celltype', method='wilcoxon', pts = True, use_raw = False, tie_correct = True)
    top_marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).iloc[0:ntop,]
    top_marker_array = np.array(top_marker_df.T)
    top_marker_list = list(set(top_marker_array.flatten().tolist()))
    return(top_marker_list)

