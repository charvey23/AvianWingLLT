"""
The purpose of this file is to read in the anatomical points defining a bird wing and to
output the appropriate dictionary to be used in MachUpX
"""
import pandas as pd
import numpy as np

# Import the MachUpX module
import machupX as MX
import json
import matplotlib.pyplot as plt

import birdwingsegmenter as bws

# ------------------------------- Import data and initialize ---------------------------------------
# compute wing segment #1
wing_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/LLT_AIAAPaper/AvianWingLLT'
                        '/2020_05_25_OrientedWings.csv')

species_data = pd.read_csv('/Users/Inman PC/Google Drive/DoctoralThesis/LLT_AIAAPaper/AvianWingLLT'
                           '/allspeciesdat.csv')
skipped_configs = []
previous_species = "initialize"

velocity_list = [5, 10, 20]  # selected test velocities
alpha_list = range(-10, 10, 1)  # selected angle of attacks
density = 1.225
dyn_vis = 1.81E-5


def round_high(number, base=5E4):
    return int(base * round(float(number)/base))


# ------------------------------- Iterate through each wing shape ---------------------------------------
# loops through each wing configuration in the data sheet by row: Currently only goes through first 10

for x in range(0, 10):
    # define the current wing name
    curr_wing_name = wing_data["species"][x] + "_WingID" + wing_data["WingID"][x] + wing_data["TestID"][
        x] + "_Frame" + str(wing_data["frameID"][x])

    bird_cg = [0, 0, 0]

    # --------------------------------------------------------------------------------------------
    # define the current points relating the each numerical point
    # NOTE: as required subtract the adjust CG of the aircraft from each pt here
    # this ensures that the twist and sweep occur properly

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
        continue
    else:
        # ---------------------- Body specific data ---------------------------------
        # only update if the species has changed - saves time
        curr_species = wing_data["species"][x]

        if previous_species != curr_species:
            species_series = species_data.species
            species_list = species_series.values.tolist()
            curr_index = species_list.index(curr_species)

            bird_weight = species_data.mass[curr_index]*9.81  # N
            body_len = species_data.l[curr_index]  # m
            body_root_airfoil = species_data.body_root_airfoil[curr_index]
            body_tip_airfoil = species_data.body_tip_airfoil[curr_index]
            inner_airfoil = species_data.proximal_airfoil[curr_index]
            mid_airfoil = species_data.mid_airfoil[curr_index]
            outer_airfoil = species_data.distal_airfoil[curr_index]
            w_2 = 0.5*species_data.w[curr_index]  # m

        previous_species = curr_species

        quarter_chord, full_chord, segment_span, full_span_frac, full_sweep, full_dihedral, full_twist = \
            bws.geom_def(curr_le, curr_te)

        # replace the placeholder airfoils with the true airfoil names
        airfoil_list_updated = [inner_airfoil if wd == "InnerAirfoil" else wd for wd in airfoil_list]
        airfoil_list_updated = [mid_airfoil if wd == "MidAirfoil" else wd for wd in airfoil_list_updated]
        airfoil_list_updated = [outer_airfoil if wd == "OuterAirfoil" else wd for wd in airfoil_list_updated]

        # calculate the average chord of each segment
        all_chord = [body_len] + full_chord
        mean_chord = [0.0]*6

        for checkRe in range(0, 6):
            mean_chord[checkRe] = 0.5 * (all_chord[checkRe + 1] + all_chord[checkRe])


# ---------------------------------------- Loop through the test velocities --------------------------------------
        for v in range(0, 3):
            test_v = velocity_list[v]

            # airfoil for each segment from proximal to distal
            all_airfoils = [body_root_airfoil] + [body_tip_airfoil] + airfoil_list_updated
            # calculate the average Re of each segment
            all_chord = [body_len] + full_chord

            # rounds the average Re section to the nearest 1E4
            Re_section = density*test_v*np.array(mean_chord)/dyn_vis
            # set both body tip Re and wing root Re as based on the wing root chord
            Re_section = np.insert(Re_section, 1, Re_section[0])
            Re_section = np.insert(Re_section, 2, (density * test_v * all_chord[1] / dyn_vis))

            # Rounds the Re as appropriate
            for re in range(0, 8):
                if Re_section[re] > 1E5:
                    Re_section[re] = round_high(Re_section[re])  # rounds to nearest 5E4 above 1E5
                else:
                    if Re_section[re] < 1E4:
                        Re_section[re] = 1E4  # ensures that 1E4 is the lowest XFoil data that is used
                    else:
                        Re_section[re] = round(Re_section[re], -4)  # rounds to nearest 1E4 below 1E5

            # create the airfoil dictionary component
            # curr_airfoil_dict = bws.build_airfoil_dict(Re_section, all_airfoils)

# ---------------------------------------- Create the wing dictionary --------------------------------------
            root_chord = full_chord[0]

            body_dict = bws.body_segment(body_root_airfoil, body_tip_airfoil, w_2, body_len, root_chord)
            # commented one allows the wing to be created in one segment - airfoil interpolator problem
            # curr_wing_dict = bws.create_smooth_wing_dict(bird_cg, bird_weight, segment_span, quarter_chord[0, :], full_span_frac,
            #                                              full_chord, full_twist, full_sweep, full_dihedral,
            #                                              body_dict, airfoil_list)
            curr_wing_dict = bws.create_wing_dict(bird_cg, bird_weight, segment_span, quarter_chord[0, :], full_chord,
                                                  full_twist, full_sweep, full_dihedral, body_dict, airfoil_list)

# ---------------------------------------- Set Flight Conditions --------------------------------------

            test_aoa = 0
            test_beta = 0

            if __name__ == "__main__":
                input_file = {
                    "tag": "Aerodynamic properties of simplified bird wing",
                    "run": {
                        "display_wireframe": {"show_legend": True},
                        "forces": {"non_dimensional": True},
                        "aero_derivatives": {},
                        "distributions": {
                            "make_plots": {"chord"}
                        }},
                    "solver": {
                        "type": "linear"},
                    "units": "English",
                    "scene": {
                        "atmosphere": {},
                        "aircraft": {
                            curr_wing_name: {
                                "file": curr_wing_dict,
                                "state": {
                                    "type": "aerodynamic",
                                    "velocity": test_v,
                                    "alpha": test_aoa,
                                    "beta": test_beta}, }}}
                }

                # ---------------------------------------- Analyze output --------------------------------------
                # Initialize Scene object.
                my_scene = MX.Scene(input_file)

                my_scene.display_wireframe(show_vortices=True, show_legend=False)

                my_scene.export_stl(filename="test.stl")
                curr_dist = my_scene.distributions()

                x = curr_dist[curr_wing_name]["segment5_left"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment4_left"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment3_left"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment2_left"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment1_left"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment1_right"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment2_right"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment3_right"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment4_right"]["cpy"] +\
                    curr_dist[curr_wing_name]["segment5_right"]["cpy"]

                y = curr_dist[curr_wing_name]["segment5_left"]["cpx"] +\
                    curr_dist[curr_wing_name]["segment4_left"]["cpx"] +\
                    curr_dist[curr_wing_name]["segment3_left"]["cpx"] +\
                    curr_dist[curr_wing_name]["segment2_left"]["cpx"] +\
                    curr_dist[curr_wing_name]["segment1_left"]["cpx"] +\
                    curr_dist[curr_wing_name]["segment1_right"]["cpx"] + \
                    curr_dist[curr_wing_name]["segment2_right"]["cpx"] + \
                    curr_dist[curr_wing_name]["segment3_right"]["cpx"] + \
                    curr_dist[curr_wing_name]["segment4_right"]["cpx"] + \
                    curr_dist[curr_wing_name]["segment5_right"]["cpx"]

                y2 = curr_dist[curr_wing_name]["segment5_left"]["cpz"] +\
                    curr_dist[curr_wing_name]["segment4_left"]["cpz"] +\
                    curr_dist[curr_wing_name]["segment3_left"]["cpz"] +\
                    curr_dist[curr_wing_name]["segment2_left"]["cpz"] +\
                    curr_dist[curr_wing_name]["segment1_left"]["cpz"] +\
                    curr_dist[curr_wing_name]["segment1_right"]["cpz"] + \
                    curr_dist[curr_wing_name]["segment2_right"]["cpz"] + \
                    curr_dist[curr_wing_name]["segment3_right"]["cpz"] + \
                    curr_dist[curr_wing_name]["segment4_right"]["cpz"] + \
                    curr_dist[curr_wing_name]["segment5_right"]["cpz"]


                # x = curr_dist[curr_wing_name]["main_wing_right"]["cpy"]
                #
                # y = curr_dist[curr_wing_name]["main_wing_right"]["cpx"]
                #
                # y2 = curr_dist[curr_wing_name]["main_wing_right"]["cpz"]

                x_c4 = [quarter_chord[0][1], quarter_chord[1][1], quarter_chord[2][1],
                        quarter_chord[3][1], quarter_chord[4][1]] + (w_2-quarter_chord[0][1])

                z_c4 = [quarter_chord[0][2], quarter_chord[1][2],
                        quarter_chord[2][2], quarter_chord[3][2],
                        quarter_chord[4][2]]

                plt.plot(x, y2, x_c4, z_c4, 'bo')
                plt.show()

                y_c4 = [quarter_chord[0][0], quarter_chord[1][0],
                        quarter_chord[2][0], quarter_chord[3][0],
                        quarter_chord[4][0]]
                plt.plot(x, y, x_c4, y_c4, 'ro')
                plt.show()

                FM_results = my_scene.solve_forces(dimensional=True, non_dimensional=False, verbose=True)
                print(json.dumps(FM_results[curr_wing_name]["total"], indent=4))
