import os
from jddb.file_repo import FileRepo
from jddb.processor import Shot, Signal
import numpy as np
import Feature_extraction
import json

config_file = "config.json"
with open(config_file, "r") as json_file:
    file_paths = json.load(json_file)
file_repo_hdf5 = FileRepo(os.path.join(file_paths['hdf5_path'], "$shot_2$00"))
file_repo_feature = FileRepo(os.path.join(file_paths['feature_path'], "$shot_2$00"))


shot_list_path = file_paths['shot_list_path']
dis_all = np.load(shot_list_path + os.sep + "DisruptAll.npy")
non_all = np.load(shot_list_path + os.sep + "NondisruptAll.npy")
shot_list = list(np.hstack((dis_all, non_all)))
# shot_list = [92919]
if not os.path.exists(file_paths['feature_path']):
    os.makedirs(file_paths['feature_path'])
# 各种tag
MNN_list = [
    r"DS-EMD-MP:NPOL-01", r"DS-EMD-MP:NPOL-02", r"DS-EMD-MP:NPOL-03", r"DS-EMD-MP:NPOL-04",
    r"DS-EMD-MP:NPOL-05", r"DS-EMD-MP:NPOL-06", r"DS-EMD-MP:NPOL-07",
    r"DS-EMD-MP:NPOL-09", r"DS-EMD-MP:NPOL-10"
]

LM_list = [
    r"DS-EMD-MP:NPOL-02", r"DS-EMD-MP:NPOL-07", r"DS-EMD-MP:NPOL-04", r"DS-EMD-MP:NPOL-09"
]  # 0, 101

Mirnov_list = [
    r"DS-EMD-MP:MPOL-01",
    r"DS-EMD-MP:MPOL-03", r"DS-EMD-MP:MPOL-04",
    r"DS-EMD-MP:MPOL-05", r"DS-EMD-MP:MPOL-06", r"DS-EMD-MP:MPOL-07", r"DS-EMD-MP:MPOL-08",
    r"DS-EMD-MP:MPOL-09", r"DS-EMD-MP:MPOL-10", r"DS-EMD-MP:MPOL-11", r"DS-EMD-MP:MPOL-12",
    r"DS-EMD-MP:MPOL-13", r"DS-EMD-MP:MPOL-14", r"DS-EMD-MP:MPOL-15", r"DS-EMD-MP:MPOL-16",
    r"DS-EMD-MP:MPOL-17", r"DS-EMD-MP:MPOL-18"

]
sxr_list = [
    r"DS-SXR-SXA:SX05", r"DS-SXR-SXA:SX06",
    r"DS-SXR-SXA:SX09", r"DS-SXR-SXA:SX10", r"DS-SXR-SXA:SX11", r"DS-SXR-SXA:SX12",
    r"DS-SXR-SXA:SX13", r"DS-SXR-SXA:SX14", r"DS-SXR-SXA:SX16",
    r"DS-SXR-SXA:SX17", r"DS-SXR-SXA:SX19"
]
xuv_list = [
    r"DS-BM-AB:BOLD01", r"DS-BM-AB:BOLD02", r"DS-BM-AB:BOLD03",
    r"DS-BM-AB:BOLD04", r"DS-BM-AB:BOLD05", r"DS-BM-AB:BOLD06",
    r"DS-BM-AB:BOLD07", r"DS-BM-AB:BOLD08", r"DS-BM-AB:BOLD09",
    r"DS-BM-AB:BOLD10", r"DS-BM-AB:BOLD11", r"DS-BM-AB:BOLD12",
    r"DS-BM-AB:BOLD14", r"DS-BM-AB:BOLD15"
]
xuv_edge_list = [r"DS-BM-AB:BOLD01", r"DS-BM-AB:BOLD02", r"DS-BM-AB:BOLD03",
                 r"DS-BM-AB:BOLD14", r"DS-BM-AB:BOLD15"]
basic_list = [
    r"CCO-LFB:LFEX-IP", r"CCO-LFB:LFBBT", r"CCO-LFB:LFDH", r"CCO-LFB:LFDV",
    r"CCO-DF:DENSITY1", "DS-EMD-ROG:VL-FILTER"
]
label_list = ["DownTime", "StartTime", "IsDisrupt"]
attribute_list = ["SampleRate", "StartTime"]