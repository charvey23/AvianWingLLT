"""
The purpose of this file is to read in the output data from bird wing shape LLT analysis and compute the key parameters
"""

import json
import pandas as pd
import os
from datetime import date
import matplotlib.pyplot as plt

# ---------------- Read in the file of all converged data points -------
dat_wingcon = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/LLT_AIAAPaper/AvianWingLLT'
                          '/List_ConvergedWings.csv')

# ---------------- Initialize lists and results -------
results_file = []
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
# ---------------- Compile all data from converged data points -------
for x in range(0, len(dat_wingcon.index)):
    wing_name_curr = dat_wingcon["species"][x] + "_WingID" + dat_wingcon["WingID"][x] + dat_wingcon["TestID"][x] + \
                     "_Frame" + str(dat_wingcon["FrameID"][x])
    filename_curr = wing_name_curr + "_U" + str(dat_wingcon["U"][x]) + "_alpha" + str(dat_wingcon["alpha"][x])

    abs_res_path = os.path.join(script_dir, "DataConverged\\" + str(filename_curr) + "_results.json")
    abs_dist_path = os.path.join(script_dir, "DataConverged\\" + str(filename_curr) + "_dist.json")

    with open(abs_res_path) as f1:
        dat_res_curr = json.load(f1)

    results_file.append([dat_wingcon["species"][x], dat_wingcon["WingID"][x],
                         dat_wingcon["TestID"][x], dat_wingcon["FrameID"][x],
                         dat_wingcon["elbow"][x], dat_wingcon["manus"][x],
                         dat_wingcon["ref_S"][x], dat_wingcon["ref_l_long"][x], dat_wingcon["ref_l_lat"][x],
                         dat_wingcon["len_tip"][x],
                         float(dat_wingcon["sweep_tip"][x].replace("[", "").replace("]", "")),
                         float(dat_wingcon["dihedral_tip"][x].replace("[", "").replace("]", "")),
                         float(dat_wingcon["sweep_tip"][x].replace("[", "").replace("]", "")),
                         dat_wingcon["alpha"][x], dat_wingcon["U"][x], dat_wingcon["build_error"][x],
                         dat_res_curr[wing_name_curr]['total']['CL'],
                         dat_res_curr[wing_name_curr]['total']['CD'],
                         dat_res_curr[wing_name_curr]['total']['Cm_w'],
                         dat_res_curr[wing_name_curr]['total']['FL'],
                         dat_res_curr[wing_name_curr]['total']['FD'],
                         dat_res_curr[wing_name_curr]['total']['My_w']])

# ------------------------------ Compute stability derivatives ------------------------------------

file_res = pd.DataFrame(results_file)
file_res.columns = ["species", "WingID", "TestID", "FrameID", "elbow", "manus",
                    "ref_S", "ref_l_long", "ref_l_lat", "len_tip", "sweep_tip", "dihedral_tip", "twist_tip",
                    "alpha", "U", "build_error", "CL", "CD", "Cm", "FL", "FD", "My"]

today = date.today()
date_adj = today.strftime("%Y_%m_%d")
file_res.to_csv(str(date_adj) + '_CompiledResults.csv', index=False)

