"""
The purpose of this file is to read in the anatomical points defining a bird wing and to
output the appropriate dictionary to be used in MachUpX
"""
import pandas as pd
import numpy as np
import machupX as MX
import matplotlib.pyplot as plt
import birdwingsegmenter as bws
import airfoil_db as adb
import os
import json

# ------------------------------- Import data and initialize ---------------------------------------
# compute wing segment #1
wing_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/LLT_AIAAPaper/AvianWingLLT'
                        '/2020_05_25_OrientedWings.csv')
# limit the data to 0 shoulder sweep and 0 shoulder dihedral for now
error_file = []
converged_file = []
converged = 1
# VRP is defined as the center location between the two humeral heads
species_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/LLT_AIAAPaper/AvianWingLLT/allspeciesdat.csv')
skipped_configs = []
skipped_reason = []
previous_species = "initialize"
all_gull_airfoils = []
all_gull_body = []
all_gull_wing_dicts = []
velocity_list = [10]  # selected test velocities
alpha_list = np.arange(-5, 5, 0.5)  # selected angle of attacks
density = 1.225
dyn_vis = 1.81E-5


def round_high(number, base=5E4):
    return int(base * round(float(number)/base))


# ------------------------------- Iterate through each wing shape ---------------------------------------
# loops through each wing configuration in the data sheet by row: Currently only goes through first 10
wt_wings = [42666, 52686, 52695, 20275, 20114, 86, 52586, 42567, 42666]
# range(0, 68977)
for x in wt_wings:
    # define the current wing name
    curr_wing_name = wing_data["species"][x] + "_WingID" + wing_data["WingID"][x] + wing_data["TestID"][
        x] + "_Frame" + str(wing_data["frameID"][x])
    curr_elbow = wing_data["elbow"][x]
    curr_manus = wing_data["manus"][x]

    # only investigate the in vivo gliding ranges
    if curr_elbow < 85:
        continue
    if curr_manus < 105:
        continue
    # only investigate no sweep and dihedral at shoulder joint for this study
    if wing_data["sweep"][x] != 0:
        continue
    if wing_data["dihedral"][x] != 0:
        continue
    # --------------------------------------------------------------------------------------------
    # define the current points relating the each numerical point
    curr_joints = np.array([[wing_data["Pt2X"][x], wing_data["Pt2Y"][x], wing_data["Pt2Z"][x]],
                            [wing_data["Pt3X"][x], wing_data["Pt3Y"][x], wing_data["Pt3Z"][x]],
                            [wing_data["Pt4X"][x], wing_data["Pt4Y"][x], wing_data["Pt4Z"][x]]])

    curr_pts = np.array([[wing_data["Pt6X"][x], wing_data["Pt6Y"][x], wing_data["Pt6Z"][x]],
                        [wing_data["Pt10X"][x], wing_data["Pt10Y"][x], wing_data["Pt10Z"][x]],
                        [wing_data["Pt7X"][x], wing_data["Pt7Y"][x], wing_data["Pt7Z"][x]],
                        [wing_data["Pt9X"][x], wing_data["Pt9Y"][x], wing_data["Pt9Z"][x]],
                        [wing_data["Pt8X"][x], wing_data["Pt8Y"][x], wing_data["Pt8Z"][x]],
                        [wing_data["Pt11X"][x], wing_data["Pt11Y"][x], wing_data["Pt11Z"][x]],
                        [wing_data["Pt12X"][x], wing_data["Pt12Y"][x], wing_data["Pt12Z"][x]]])
    curr_edges = ["LE", "TE", "LE", "TE", "NA", "TE", "LE"]
    # The number of points that were digitized on the wing - the last two pts should be on the LE and TE of the root
    no_pts = 7

    # outputs the leading and trailing edge and airfoil of each segment
    curr_le, curr_te, airfoil_list = bws.segmenter(curr_pts, curr_edges, curr_joints, no_pts)

    if curr_le is None:
        print(curr_wing_name + ' is too tucked. Continuing to next configuration.')
        skipped_configs.append(curr_wing_name)
        skipped_reason.append("tucked")
        continue
    else:
        # ---------------------- Bird specific data ---------------------------------
        # only update if the species has changed
        not_prev = False
        curr_species = wing_data["species"][x]

        if previous_species != curr_species:
            not_prev = True
            species_series = species_data.species
            species_list = species_series.values.tolist()
            # find the index that corresponds to the current species
            curr_index = species_list.index(curr_species)
            # define bird specific geometry
            bird_weight = species_data.mass[curr_index]*9.81  # N
            body_len = species_data.l[curr_index]  # m
            w_2 = 0.5 * species_data.w[curr_index]  # m
            wsk_2 = 0.5 * species_data.w_sk[curr_index]  # m

            # define bird specific airfoils
            body_root_airfoil = species_data.body_root_airfoil[curr_index]
            inner_airfoil = species_data.proximal_airfoil[curr_index]
            mid_airfoil = species_data.mid_airfoil[curr_index]
            outer_airfoil = species_data.distal_airfoil[curr_index]
            # define CG relative to the VRP
            bird_cg = [float(species_data.CG_x[curr_index]),
                       float(species_data.CG_y[curr_index]),
                       float(species_data.CG_z[curr_index])]

        previous_species = curr_species
        # ---------------------- Wing geometry data ---------------------------------
        # determine the appropriate geometry along the wingspan
        gam = 0.85
        body_gam = [(wsk_2 / w_2), 0.1 + (wsk_2 / w_2), 0.9]
        quarter_chord, full_chord, airfoil_list, segment_span, full_span_frac, dis_span_frac, \
        wing_sweep, wing_dihedral, full_twist, body_span_frac, true_body_w2, body_sweep, body_dihedral \
            = bws.geom_def(curr_le, curr_te, gam, body_gam, w_2, no_pts, airfoil_list)

        # replace the placeholder airfoils with the true airfoil names
        airfoil_list_updated = [inner_airfoil if wd == "InnerAirfoil" else wd for wd in airfoil_list]
        airfoil_list_updated = [mid_airfoil if wd == "MidAirfoil" else wd for wd in airfoil_list_updated]
        airfoil_list_updated = [outer_airfoil if wd == "OuterAirfoil" else wd for wd in airfoil_list_updated]

        all_chord = np.insert(full_chord, 0, body_len)
        # NOTE: body airfoil is added a second time because the edge of the body airfoil will have a different Reynolds
        all_airfoils = [body_root_airfoil] + [body_root_airfoil] + airfoil_list_updated
        # NOTE: root chord is added a second time because the edge of the body airfoil will have a different Reynolds
        all_chord = np.insert(all_chord, 1, full_chord[0])

        # -------------------------------------- Loop through the test velocities --------------------------------------
        for v in range(0, len(velocity_list)):
            test_v = velocity_list[v]

            # calculates the Re at each section slice
            Re_section = density*test_v*np.array(all_chord)/dyn_vis
            final_airfoil_names = ['0']*len(Re_section)
            # Round the Reynolds number
            for re in range(0, len(Re_section)):
                if Re_section[re] > 1E5:
                    Re_section[re] = round_high(Re_section[re])  # rounds to nearest 5E4 above 1E5
                else:
                    if Re_section[re] < 1E4:
                        Re_section[re] = 1E4  # ensures that 1E4 is the lowest XFoil data that is used
                    else:
                        Re_section[re] = round(Re_section[re], -4)  # rounds to nearest 1E4 below 1E5
                # acquire the final names
                final_airfoil_names[re] = str(all_airfoils[re]).lower() + "_Re" + str(int(Re_section[re]))

            # ----------------------------------- Create the required dictionaries -------------------------------------

            # 1. create the airfoil dictionary
            curr_airfoil_dict = bws.build_airfoil_dict(Re_section, all_airfoils)

            # 2. create the body dictionary
            curr_body_dict, area_body = bws.build_body_dict(final_airfoil_names[0:3], true_body_w2, body_len, full_chord[0],
                                                 body_span_frac, body_dihedral, wing_dihedral[0], body_sweep, wing_sweep[0])

            # 3. create the full wing dictionary
            curr_wing_dict = bws.build_smooth_wing_dict(bird_cg, bird_weight, segment_span, full_span_frac, full_chord,
                                                        full_twist, wing_sweep, wing_dihedral, dis_span_frac,
                                                        curr_body_dict, curr_airfoil_dict, final_airfoil_names)

            # ---------------------------------------- Set Flight Conditions --------------------------------------
            for alpha in alpha_list:
                test_aoa = alpha
                test_beta = 0

                if __name__ == "__main__":
                    input_file = {
                        "tag": "Aerodynamic properties of simplified bird wing",
                        "run": {
                            "display_wireframe": {"show_legend": True},
                            "solve_forces": {"non_dimensional": True},
                            "distributions": {}},
                        "solver": {
                            "type": "nonlinear",
                            "relaxation": 0.01,
                            "max_iterations": 10000,
                            'convergence': 1e-3
                        },
                        "units": "SI",
                        "scene": {
                            "atmosphere": {},
                            "aircraft": {
                                curr_wing_name: {
                                    "file": curr_wing_dict,
                                    "state": {
                                        "type": "aerodynamic",
                                        "velocity": test_v,
                                        "alpha": test_aoa}, }}}
                    }

                    # ---------------------------------------- Prepare scene --------------------------------------
                    # Initialize Scene object.
                    my_scene = MX.Scene(input_file)
                    # set the errors to only warn when iterating
                    my_scene.set_err_state(not_converged="raise", database_bounds="raise")

                    # Only use the following line if printing for wind tunnel analyses
                    # my_scene.export_dxf(number_guide_curves=10, section_resolution=300)

                    # save the final reference geometry used in the analysis
                    final_geom = my_scene.get_aircraft_reference_geometry()

                    # ---------------------------------------- Solve forces --------------------------------------
                    try:
                        # get relative path for the loads file
                        script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
                        results_file = curr_wing_name + "_U" + str(test_v) + "_AOA" + str(test_aoa) + "_results.json"
                        abs_res_path = os.path.join(script_dir, "DataConverged\\" + str(results_file))

                        # solve and save the loads
                        results = my_scene.solve_forces(dimensional=True, non_dimensional=True,
                                                        verbose=True, report_by_segment=True, filename=abs_res_path)

                        # save the distributions
                        dist_file = curr_wing_name + "_U" + str(test_v) + "_AOA" + str(test_aoa) + "_dist.json"
                        abs_dist_path = os.path.join(script_dir, "DataConverged\\" + str(dist_file))
                        curr_dist = my_scene.distributions(filename=abs_dist_path)

                    except MX.SolverNotConvergedError as err_msg:
                        converged = 0
                        err = "convergence"
                    except adb.DatabaseBoundsError as err_msg:
                        converged = 0
                        err = "bounds"
                    except UserWarning as err_msg:
                        converged = 0
                        err = err_msg
                    except Exception as err_msg:
                        converged = 0
                        err = err_msg
                    except:
                        converged = 0
                        err = "unknown"
                    else:
                        converged = 1
                        err = "None"

                    if converged == 0:
                        # Uncoverged case
                        print(curr_wing_name, "at alpha =", test_aoa, "did not save because of a", err, "error")
                        # save the wing info to a file
                        error_file.append([wing_data["species"][x], wing_data["WingID"][x],
                                           wing_data["TestID"][x], wing_data["frameID"][x],
                                           curr_elbow, curr_manus,
                                           final_geom[0], final_geom[1], final_geom[2],
                                           test_aoa, test_v, err])

                        if test_aoa == 0:  # only save once per wing

                            # save the distributions for the current wing
                            script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
                            dist_file = curr_wing_name + "_dist.json"
                            abs_dist_path = os.path.join(script_dir, "DataNotConverged\\" + str(dist_file))
                            curr_dist = my_scene.distributions(filename=abs_dist_path)

                    else:
                        print(curr_wing_name, "solver converged!")

                        # verify that MachUp quarter chord matches the desired chord
                        y_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpy"]
                        x_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpx"]
                        z_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpz"]
                        x_c4 = [row[0] for row in quarter_chord]
                        y_c4 = [row[1] for row in quarter_chord] - (quarter_chord[0][1]) + w_2
                        z_c4 = [row[2] for row in quarter_chord]
                        # calculate the total maximum error on the estimated shape
                        build_error = bws.check_build_error(np.transpose(np.array([x_c4, y_c4, z_c4])), x_MX, y_MX, z_MX)

                        converged_file.append([wing_data["species"][x], wing_data["WingID"][x],
                                               wing_data["TestID"][x], wing_data["frameID"][x],
                                               curr_elbow, curr_manus,
                                               final_geom[0], final_geom[1], final_geom[2],
                                               test_aoa, test_v, max(build_error)])

                # y_body = curr_dist[curr_wing_name]["body_right"]["cpy"]
                # x_body = curr_dist[curr_wing_name]["body_right"]["cpx"]
                # z_body = curr_dist[curr_wing_name]["body_right"]["cpz"]
                #
                # x_true = [row[0] for row in curr_pts]
                # y_true = [row[1] for row in curr_pts] - curr_pts[6][1] + w_2
                # z_true = [row[2] for row in curr_pts]
                #
                # x_le = [row[0] for row in curr_le]
                # y_le = [row[1] for row in curr_le] - (curr_le[0][1]) + w_2
                # z_le = [row[2] for row in curr_le]
                #
                # x_te = [row[0] for row in curr_te]
                # y_te = [row[1] for row in curr_te] - (curr_te[0][1]) + w_2
                # z_te = [row[2] for row in curr_te]
                #
                # y = curr_dist[curr_wing_name]["main_wing_right"]["cpy"]
                # x = curr_dist[curr_wing_name]["main_wing_right"]["cpx"]
                # z = curr_dist[curr_wing_name]["main_wing_right"]["cpz"]
                #
                # x_c4 = [row[0] for row in quarter_chord]
                # y_c4 = [row[1] for row in quarter_chord] - (quarter_chord[0][1]) + w_2
                # z_c4 = [row[2] for row in quarter_chord]

                # fig = plt.figure()
                # ax = plt.axes(projection='3d')
                # ax.scatter3D(x_true, y_true, z_true, 'gray')
                # ax.scatter3D(x_c4, y_c4, z_c4, 'green')

                # plt.plot(y_body, z_body, y_MX, z_MX, y_c4, z_c4, 'bo')
                # plt.show()
                # plt.plot(y_body, x_body, y_MX, x_MX, y_c4, x_c4, 'ro')
                # plt.show()
                #
                # plt.plot(y_true, x_true, 'bo', y_c4, x_c4, 'ro')
                # plt.show()
                # print("hi")

# save the data
file_con = pd.DataFrame(converged_file)
file_con.columns = ["species", "WingID", "TestID", "FrameID", "elbow", "manus",
                    "ref_S", "ref_l_long", "ref_l_lat", "alpha", "v", "build_error"]

file_err = pd.DataFrame(error_file)
file_err.columns = ["species", "WingID", "TestID", "FrameID", "elbow", "manus",
                    "ref_S", "ref_l_long", "ref_l_lat", "alpha", "v", "error_reason"]

file_con.to_csv('List_ConvergedWings.csv', index=False)
file_err.to_csv('List_NotConvergedWings.csv', index=False)
