"""
The purpose of this file is to read in the anatomical points defining a bird wing and to
output the appropriate dictionary to be used in MachUpX
"""
import pandas as pd
import numpy as np
import machupX as MX
import json
import matplotlib.pyplot as plt
import birdwingsegmenter as bws
import scipy

# ------------------------------- Import data and initialize ---------------------------------------
# compute wing segment #1
wing_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/LLT_AIAAPaper/AvianWingLLT'
                        '/2020_05_25_OrientedWings.csv')
# limit the data to 0 shoulder sweep and 0 shoulder dihedral for now

# VRP is defined as the center location between the two humeral heads
species_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/LLT_AIAAPaper/AvianWingLLT/allspeciesdat.csv')
skipped_configs = []
skipped_reason = []
previous_species = "initialize"

velocity_list = [10, 20]  # selected test velocities
alpha_list = np.arange(-5, 5, 0.5)  # selected angle of attacks
density = 1.225
dyn_vis = 1.81E-5


def round_high(number, base=5E4):
    return int(base * round(float(number)/base))


# ------------------------------- Iterate through each wing shape ---------------------------------------
# loops through each wing configuration in the data sheet by row: Currently only goes through first 10

for x in range(52425, 52426):
    # define the current wing name
    curr_wing_name = wing_data["species"][x] + "_WingID" + wing_data["WingID"][x] + wing_data["TestID"][
        x] + "_Frame" + str(wing_data["frameID"][x])

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

    # outputs the leading and trailing edge and airfoil of each segment
    curr_le, curr_te, airfoil_list = bws.segmenter(curr_pts, curr_edges, curr_joints)

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
        body_gam = [(wsk_2 / w_2), 0.05 + (wsk_2 / w_2), 0.95]
        quarter_chord, full_chord, segment_span, full_span_frac, dis_span_frac, \
        wing_sweep, wing_dihedral, full_twist, body_span_frac, true_body_w2, body_sweep, body_dihedral \
            = bws.geom_def(curr_le, curr_te, gam, body_gam, w_2)

        # replace the placeholder airfoils with the true airfoil names
        airfoil_list_updated = [inner_airfoil if wd == "InnerAirfoil" else wd for wd in airfoil_list]
        airfoil_list_updated = [mid_airfoil if wd == "MidAirfoil" else wd for wd in airfoil_list_updated]
        airfoil_list_updated = [outer_airfoil if wd == "OuterAirfoil" else wd for wd in airfoil_list_updated]

        # calculate the average chord of each segment
        all_chord = [body_len] + full_chord
        root_chord = full_chord[0]
        # airfoil for each segment from proximal to distal
        # NOTE: body airfoil is added a second time because the edge of the body airfoil will have a different Reynolds
        all_airfoils = [body_root_airfoil] + [body_root_airfoil] + airfoil_list_updated
        # NOTE: root chord is added a second time because the edge of the body airfoil will have a different Reynolds
        all_chord = [body_len] + [root_chord] + full_chord

# ---------------------------------------- Loop through the test velocities --------------------------------------
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

# -------------------------------------- Create the required dictionaries --------------------------------------

            # 1. create the airfoil dictionary

            curr_airfoil_dict = bws.build_airfoil_dict(Re_section, all_airfoils)
            # 2. create the body dictionary
            curr_body_dict = bws.build_body_dict(final_airfoil_names[0:3], true_body_w2, body_len, root_chord,
                                                 body_span_frac, body_dihedral, wing_dihedral[0], body_sweep, wing_sweep[0])

            # 3. create the full wing dictionary
            curr_wing_dict = bws.build_smooth_wing_dict(bird_cg, bird_weight, segment_span, full_span_frac, full_chord,
                                                        full_twist, wing_sweep, wing_dihedral, dis_span_frac,
                                                        curr_body_dict, curr_airfoil_dict, final_airfoil_names)
            # curr_wing_dict = bws.build_segment_wing_dict(bird_cg, bird_weight, segment_span, quarter_chord[0, :],
            #                                              full_chord, full_twist, full_sweep, full_dihedral,
            #                                              body_dict, curr_airfoil_dict)

# ---------------------------------------- Set Flight Conditions --------------------------------------

            test_aoa = 0
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
                        "max_iterations": 50
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

                # ---------------------------------------- Analyze output --------------------------------------
                # Initialize Scene object.
                my_scene = MX.Scene(input_file)
                # my_scene.display_wireframe(show_vortices=True, show_legend=False)
                # FM_results = my_scene.solve_forces(dimensional=True, non_dimensional=True,
                #                                    verbose=True, report_by_segment=True)
                # my_scene.export_stl(filename="test.stl")
                curr_dist = my_scene.distributions()

                # put in check that computes the error between the known quarter chord and the resultant quarter chord
                # if greater than a preset tolerance then skip this wing

                # verify that MachUp quarter chord matches the desired chord

                y = curr_dist[curr_wing_name]["main_wing_right"]["cpy"]
                x = curr_dist[curr_wing_name]["main_wing_right"]["cpx"]
                z = curr_dist[curr_wing_name]["main_wing_right"]["cpz"]

                y_body = curr_dist[curr_wing_name]["body_right"]["cpy"]
                x_body = curr_dist[curr_wing_name]["body_right"]["cpx"]
                z_body = curr_dist[curr_wing_name]["body_right"]["cpz"]

                x_c4 = [quarter_chord[0][0], quarter_chord[1][0], quarter_chord[2][0], quarter_chord[3][0],
                        quarter_chord[4][0], quarter_chord[5][0]]
                y_c4 = [quarter_chord[0][1], quarter_chord[1][1], quarter_chord[2][1],
                        quarter_chord[3][1], quarter_chord[4][1], quarter_chord[5][1]] - (quarter_chord[0][1]) + w_2
                z_c4 = [quarter_chord[0][2], quarter_chord[1][2],
                        quarter_chord[2][2], quarter_chord[3][2],
                        quarter_chord[4][2], quarter_chord[5][2]]

                x_true = [curr_pts[0][0], curr_pts[1][0], curr_pts[2][0], curr_pts[3][0], curr_pts[4][0],
                          curr_pts[5][0], curr_pts[6][0]]
                y_true = [curr_pts[0][1], curr_pts[1][1], curr_pts[2][1], curr_pts[3][1], curr_pts[4][1],
                          curr_pts[5][1], curr_pts[6][1]] - curr_pts[6][1] + w_2
                z_true = [curr_pts[0][2], curr_pts[1][2], curr_pts[2][2], curr_pts[3][2], curr_pts[4][2],
                          curr_pts[5][2], curr_pts[6][2]]

                # calculate the maximum error on each point
                build_error = bws.check_build_error(np.transpose(np.array([x_c4, y_c4, z_c4])), x, y, z)

                plt.plot(y_body, z_body, y, z, y_c4, z_c4, 'bo')
                plt.show()
                plt.plot(y_body, x_body, y, x, y_c4, x_c4, 'ro')
                plt.show()

                plt.plot(y_true, x_true, 'bo', y_c4, x_c4, 'ro')
                plt.show()
                print("hi")

                # print(json.dumps(FM_results[curr_wing_name]["total"], indent=4))
