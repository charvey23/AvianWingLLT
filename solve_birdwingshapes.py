"""
The purpose of this file is to read in the anatomical points defining a bird wing and to
output the appropriate dictionary to be used in MachUpX
"""
import pandas as pd
import numpy as np
import compute_birdwingshapes as bws
import run_birdwingLLT as rbw
import csv
from datetime import date
import machupX as MX

# ------------------------------- Import data and initialize ---------------------------------------
# compute wing segment #1
wing_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/StaticStability/AvianWingLLT'
                        '/2020_05_25_OrientedWings.csv')
# limit the data to 0 shoulder sweep and 0 shoulder dihedral for now
error_file = []
converged_file = []
converged = 1
# VRP is defined as the center location between the two humeral heads
species_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/StaticStability/AvianWingLLT/allspeciesdat.csv')
skipped_configs = []
skipped_reason = []
previous_species = "initialize"
all_gull_airfoils = []
all_gull_body = []
all_gull_wing_dicts = []

# -------------------------------  Set the test conditions -------------------------------
velocity_list = [10]  # selected test velocities
alpha_list = np.arange(-10.0, 10.0, 1.0)  # selected angle of attacks
density = 1.225
dyn_vis = 1.81E-5
kin_vis = dyn_vis/density

# ------------------------------- Iterate through each wing shape ---------------------------------------
# loops through each wing configuration in the data sheet by row: Currently only goes through first 10
wt_wings = [52686, 52695, 52770, 42597, 52648, 52586, 42642, 42567, 42666]

for x in range(0, len(wing_data.index)):
    # define the current wing name
    curr_wing_name = wing_data["species"][x] + "_WingID" + wing_data["WingID"][x] + wing_data["TestID"][
        x] + "_Frame" + str(wing_data["frameID"][x])
    curr_elbow = wing_data["elbow"][x]
    curr_manus = wing_data["manus"][x]
    print('---------------------------------------')
    print('Computing wing shape for: %s' % curr_wing_name)
    print('---------------------------------------')

    # ----------------------------- Remove wings that will not be investigated ---------------------------------
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
        gam = 0.95
        # defines the ratio use at the body. 1st entry is the final place for 0 sweep and 0 dihedral,
        # between 1st and 2nd is the linear region, 2nd and 3rd is constant dihedral and sweep,
        # 3rd to end is another linear region that ends at the root qualities
        body_gam = [(wsk_2 / w_2), 0.05 + (wsk_2 / w_2), 0.90]
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

        full_model_length = wsk_2 + max(quarter_chord[:, 1])

        # -------------------------------------- Loop through the test velocities --------------------------------------
        for v in range(0, len(velocity_list)):
            test_v = velocity_list[v]

            # calculates the Re at each section slice
            Re_section = density*test_v*np.array(all_chord)/dyn_vis
            final_airfoil_names = ['0']*len(Re_section)
            # Round the Reynolds number
            for re in range(0, len(Re_section)):
                if Re_section[re] > 1E5:
                    Re_section[re] = bws.round_high(Re_section[re])  # rounds to nearest 5E4 above 1E5
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
                message_str = "Current wing: %s Current Angle of attack: %d" \
                              % (curr_wing_name, alpha)
                print(message_str)
                print('---------------------------------------')
                test_aoa = alpha
                test_beta = 0

                # set the solver information
                max_it = 1E2
                tol_con = 1E-8
                relax = 1

                if __name__ == "__main__":

                    message_str = "1st Attempt: maximum iterations = %d and relaxation= %.2f" % (max_it,  relax)
                    print(message_str)
                    print('---------------------------------------')
                    # create input file and try to solve
                    converged, err, final_geom, \
                    mac_curr, results, curr_dist = rbw.run_machupx(curr_wing_name, test_v, test_aoa, density, kin_vis,
                                                                   curr_wing_dict, max_it, tol_con, relax)

                    # adjust the relaxation factor and iterations if did not converge - don't bother with the positives
                    if converged == 0 and alpha < 0:
                        max_it = 1E3
                        tol_con = 1E-8
                        relax = 0.8

                        message_str = "2nd Attempt: maximum iterations = %d and relaxation= %.2f" % (max_it,  relax)
                        print('---------------------------------------')
                        print(message_str)
                        print('---------------------------------------')
                        # create input file and try to solve
                        converged, err, final_geom, \
                        mac_curr, results, curr_dist = rbw.run_machupx(curr_wing_name, test_v, test_aoa,
                                                                       density, kin_vis, curr_wing_dict,
                                                                       max_it, tol_con, relax)
                    # adjust the relaxation factor and iterations if did not converge - don't bother with the positives
                    if converged == 0 and alpha < 0:
                        max_it = 1E3
                        tol_con = 1E-8
                        relax = 0.5

                        message_str = "3rd Attempt: maximum iterations = %d and relaxation= %.2f" % (max_it,  relax)
                        print('---------------------------------------')
                        print(message_str)
                        print('---------------------------------------')
                        # create input file and try to solve
                        converged, err, final_geom, \
                        mac_curr, results, curr_dist = rbw.run_machupx(curr_wing_name, test_v, test_aoa,
                                                                       density, kin_vis, curr_wing_dict,
                                                                       max_it, tol_con, relax)

                    # adjust the relaxation factor and iterations if did not converge
                    if converged == 0:
                        max_it = 1E4
                        tol_con = 1E-8
                        relax = 0.01
                        message_str = "4th Attempt: maximum iterations = %d and relaxation= %.2f" % (max_it,  relax)
                        print('---------------------------------------')
                        print(message_str)
                        print('---------------------------------------')
                        # create input file and try to solve
                        converged, err, final_geom, \
                        mac_curr, results, curr_dist = rbw.run_machupx(curr_wing_name, test_v, test_aoa,
                                                                       density, kin_vis, curr_wing_dict,
                                                                       max_it, tol_con, relax)

                    # save current date
                    today = date.today()
                    date_adj = today.strftime("%Y_%m_%d")

                    # ---------------------------------------- Save information ------------------------------------
                    print('---------------------------------------')
                    print('Saving data')
                    print('---------------------------------------')
                    if converged == 0:
                        # Uncoverged case
                        print(curr_wing_name, "at alpha =", test_aoa, "did not save because of a", err, "error")
                        # save the wing info to a file
                        with open('List_NotConvergedWings.csv', 'a', newline="") as err_file:
                            writer = csv.writer(err_file)
                            writer.writerow([wing_data["species"][x], wing_data["WingID"][x],
                                            wing_data["TestID"][x], wing_data["frameID"][x],
                                            curr_elbow, curr_manus, test_aoa, test_v, err, date_adj,
                                            full_model_length,
                                             wing_sweep[-1:][0], wing_dihedral[-1:][0], full_twist[-1:][0], relax])

                    else:
                        print(curr_wing_name, "at alpha =", test_aoa, "converged!")

                        # verify that MachUp quarter chord matches the desired chord
                        y_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpy"]
                        x_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpx"]
                        z_MX = curr_dist[curr_wing_name]["main_wing_right"]["cpz"]
                        x_c4 = [row[0] for row in quarter_chord]
                        y_c4 = [row[1] for row in quarter_chord] - (quarter_chord[0][1]) + w_2
                        z_c4 = [row[2] for row in quarter_chord]

                        # calculate the total maximum error on the estimated shape
                        build_error = bws.check_build_error(np.transpose(np.array([x_c4, y_c4, z_c4])), x_MX, y_MX, z_MX)
                        with open('List_ConvergedWings.csv', 'a', newline="") as con_file:
                            writer = csv.writer(con_file)
                            writer.writerow([wing_data["species"][x], wing_data["WingID"][x],
                                             wing_data["TestID"][x], wing_data["frameID"][x],
                                             curr_elbow, curr_manus, test_aoa, test_v, max(build_error), date_adj,
                                             final_geom[0], final_geom[1], final_geom[2],
                                             mac_curr[curr_wing_name]["length"],
                                             full_model_length,
                                             wing_sweep[-1:][0], wing_dihedral[-1:][0], full_twist[-1:][0], relax,
                                            results[curr_wing_name]['total']['CL'],
                                            results[curr_wing_name]['total']['CD'],
                                            results[curr_wing_name]['total']['Cm_w'],
                                            results[curr_wing_name]['total']['FL'],
                                            results[curr_wing_name]['total']['FD'],
                                            results[curr_wing_name]['total']['My_w']])

