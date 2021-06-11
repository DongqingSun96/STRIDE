library(optparse)
library(Seurat)

option_list = list(
  make_option(c("--sc_count_file"), type = "character", 
              action = "store", help = "The location of single cell count matrix. It should be '.h5' format. "
  ),
  make_option(c("--sc_anno_file"), type = "character", 
              action = "store", help = "Location of the single-cell celltype annotation file. 
              The file should be a tab-separated plain-text file without header. 
              The first column should be the cell name, and the second column should be the corresponding celltype labels. "
  ),
  make_option(c("--out_dir"), type = "character", 
              action = "store", help = "Path to the directory where the result file shall be stored. "
  ),
  make_option(c("--out_prefix"), type = "character", 
              action = "store", help = "Prefix of output files. "
  )
)

argue = parse_args(OptionParser(option_list = option_list, usage = "Find marker genes for each cluster. "))

SCMarkerFind <- function(sc_count_file, sc_anno_file, out_dir, out_prefix = "STRIDE.scRNA"){
  inputMat = Read10X_h5(sc_count_file)
  meta.data = read.table(sc_anno_file, header = FALSE, row.names = 1, sep = "\t")
  colnames(meta.data) = "Celltype"
  SeuratObj <- CreateSeuratObject(counts = inputMat, project = out_prefix,
                                  min.cells = 10, min.features = 200, meta.data = meta.data)
  SeuratObj <- NormalizeData(object = SeuratObj, normalization.method = "LogNormalize", scale.factor = 10000)
  SeuratObj <- FindVariableFeatures(object = SeuratObj, selection.method = "vst", nfeatures = 2000)
  SeuratObj <- ScaleData(object = SeuratObj, vars.to.regress = "nCount_RNA")

  markers.celltype <- presto:::wilcoxauc.Seurat(X = SeuratObj, group_by = 'Celltype', assay = 'data')
  markers.celltype <- markers.celltype[markers.celltype$padj < 1E-5 & markers.celltype$logFC > 0.25, ]
  
  markers.celltype.list = split(markers.celltype$feature, markers.celltype$group)
  markers.celltype.len = sapply(markers.celltype.list, length)
  if(min(markers.celltype.len) > 150) {
    markers.per.celltype.num = min(markers.celltype.len)
  }else{
    markers.per.celltype.num = 150
  }
  markers.celltype.top.list = lapply(markers.celltype.list, function(x){
    if(length(x) > markers.per.celltype.num){
      return(x[1:min(markers.celltype.len)])
    } else{
      return(x)
    }
  })
  markers.celltype.top.unique = unique(unlist(markers.celltype.top.list))
  marker_file = file.path(out_dir, paste0(SeuratObj@project.name, "_markers_top.txt"))
  write.table(markers.celltype.top.unique, marker_file, quote = FALSE, col.names = FALSE, row.names = FALSE)
}

sc_count_file = argue$sc_count_file
sc_anno_file = argue$sc_anno_file
out_dir = argue$out_dir
out_prefix = argue$out_prefix

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}
SCMarkerFind(sc_count_file, sc_anno_file, out_dir, out_prefix)
