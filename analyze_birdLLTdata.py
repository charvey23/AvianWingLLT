"""
The purpose of this file is to read in the output data from bird wing shape LLT analysis and compute the key parameters
"""

import json
import pandas as pd
import os
from datetime import date
import machupX as MX
import numpy as np

# ---------------- Initialize lists and results -------
wing_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/StaticStability/AvianWingLLT'
                        '/2020_05_25_OrientedWings.csv')
# subset the data appropriately
curr_wing_data = wing_data.loc[(wing_data["species"] == "lar_gla") & (wing_data["sweep"] == 0) & (wing_data["dihedral"] == 0)]

results_file = []
script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
converged_dir = os.path.join(script_dir, "DataConverged\\2020_09_04_NumericalRun")

# ---------------- Compile all data from converged data points -------
for curr_filename in os.listdir(converged_dir):

    if curr_filename.endswith("_results.json"):
        # 1. load in the results file
        file_dir = os.path.join(script_dir, "DataConverged\\2020_09_04_NumericalRun\\" + curr_filename)
        with open(file_dir) as f:
            dat_res_curr = json.load(f)

        # 2. load in the distribution file
        loc = curr_filename.find("_results")
        curr_filename_dist = curr_filename[0:(loc)] + "_dist.json"
        file_dir = os.path.join(script_dir, "DataConverged\\2020_09_04_NumericalRun\\" + curr_filename_dist)
        with open(file_dir) as f:
            dat_dist_curr = np.loadtxt(f, usecols=np.arange(2, 30))

        # 3. load in the input file
        locU = curr_filename.find("U")-1  # last index in the frame ID
        curr_wing_name = curr_filename[0:locU]
        curr_filename_input = curr_wing_name + "inputs.json"
        input_dir = os.path.join(script_dir, "InputFiles\\" + curr_filename_input)
        with open(input_dir) as f:
            dat_input_curr = json.load(f)

        # Create scene in MachUpX
        my_scene = MX.Scene(dat_input_curr)
        final_geom = my_scene.get_aircraft_reference_geometry()
        mac_curr = my_scene.MAC()

        full_model_length = max(dat_dist_curr[:, 2])

        # Split up current name into the necessary inputs
        locWID = curr_filename.find("WingID") + 6
        WingID = curr_filename[locWID:locWID+7]
        locTID = curr_filename.find("Test") + 4
        TestID = "Test" + str(curr_filename[locTID])
        locFID = curr_filename.find("Frame") + 5
        FrameID = float(curr_filename[locFID:locU])
        dat_wing = curr_wing_data[(curr_wing_data["WingID"] == WingID) & (curr_wing_data["TestID"] == TestID) &
                                        (curr_wing_data["frameID"] == FrameID)]

        results_file.append(['lar_gla', WingID, TestID, int(FrameID),
                             dat_wing.elbow.values[0], dat_wing.manus.values[0],
                             final_geom[0], final_geom[1], final_geom[2],
                             mac_curr[curr_wing_name]["length"],
                             dat_input_curr['scene']['aircraft'][curr_wing_name]['file']['wings']['main_wing']['sweep'][-1][1],
                             dat_input_curr['scene']['aircraft'][curr_wing_name]['file']['wings']['main_wing']['dihedral'][-1][1],
                             dat_input_curr['scene']['aircraft'][curr_wing_name]['file']['wings']['main_wing']['twist'][-1][1],
                             dat_input_curr['scene']['aircraft'][curr_wing_name]['state']['alpha'],
                             dat_input_curr['scene']['aircraft'][curr_wing_name]['state']['velocity'],
                             dat_res_curr[curr_wing_name]['total']['CL'],
                             dat_res_curr[curr_wing_name]['total']['CD'],
                             dat_res_curr[curr_wing_name]['total']['Cm_w'],
                             dat_res_curr[curr_wing_name]['total']['FL'],
                             dat_res_curr[curr_wing_name]['total']['FD'],
                             dat_res_curr[curr_wing_name]['total']['My_w']])

# ------------------------------ Compute stability derivatives ------------------------------------

file_res = pd.DataFrame(results_file)
file_res.columns = ["species", "WingID", "TestID", "FrameID", "elbow", "manus",
                    "ref_S", "ref_l_long", "ref_l_lat", "MAC", "sweep_tip", "dihedral_tip", "twist_tip",
                    "alpha", "U", "build_error", "CL", "CD", "Cm", "FL", "FD", "My"]

today = date.today()
date_adj = today.strftime("%Y_%m_%d")
file_res.to_csv(str(date_adj) + '_CompiledResults.csv', index=False)

#                 y_body = curr_dist[curr_wing_name]["body_right"]["cpy"]
#                 x_body = curr_dist[curr_wing_name]["body_right"]["cpx"]
#                 z_body = curr_dist[curr_wing_name]["body_right"]["cpz"]
#
#                 x_true = [row[0] for row in curr_pts]
#                 y_true = [row[1] for row in curr_pts] - curr_pts[6][1] + w_2
#                 z_true = [row[2] for row in curr_pts]
#
#                 x_le = [row[0] for row in curr_le]
#                 y_le = [row[1] for row in curr_le] - (curr_le[0][1]) + w_2
#                 z_le = [row[2] for row in curr_le]
#
#                 x_te = [row[0] for row in curr_te]
#                 y_te = [row[1] for row in curr_te] - (curr_te[0][1]) + w_2
#                 z_te = [row[2] for row in curr_te]
#
#                 y_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpy"]
#                 x_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpx"]
#                 z_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpz"]
#
#                 x_c4 = [row[0] for row in quarter_chord]
#                 y_c4 = [row[1] for row in quarter_chord] - (quarter_chord[0][1]) + w_2
#                 z_c4 = [row[2] for row in quarter_chord]
#
#                 # fig = plt.figure()
#                 # ax = plt.axes(projection='3d')
#                 # ax.scatter3D(x_true, y_true, z_true, 'gray')
#                 # ax.scatter3D(x_c4, y_c4, z_c4, 'green')
#
#                 plt.plot(y_body, z_body, y_MX, z_MX, y_c4, z_c4, 'bo')
#                 plt.show()
#                 plt.plot(y_body, x_body, y_MX, x_MX, y_c4, x_c4, 'ro')
#                 plt.show()
#
#                 plt.plot(y_true, x_true, 'bo', y_c4, x_c4, 'ro')
#                 plt.show()
#                 print("hi")
#