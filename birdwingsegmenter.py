"""
The purpose of this file is to read in the anatomical points defining a bird wing and to
output the appropriate dictionary to be used in MachUpX
"""
import numpy as np
import math
import os
import airfoil_db as adb

# --------------------------- Function definitions -------------------------------------------


def point_on_line(y_pos, pt_start, pt_end):
    direction = np.subtract(pt_end, pt_start)
    x_pos = ((direction[0] / direction[1]) * (y_pos - pt_start[1])) + pt_start[0]
    z_pos = ((direction[2] / direction[1]) * (y_pos - pt_start[1])) + pt_start[2]
    return x_pos, z_pos


def det_pseudo_pt(four_points, too_far, ind_le, ind_te):
    # pts are defined so that the last point is the further out
    # too_far should be whether or not that last point is on the "LE" or "TE"
    pseudo_y = four_points[2, 1]
    if too_far == "LE":
        ref_pt = four_points[ind_le, :]
        location = "LE"
    else:
        ref_pt = four_points[ind_te, :]
        location = "TE"

    pseudo_x, pseudo_z = point_on_line(pseudo_y, ref_pt, four_points[3, :])
    p_pt = np.array([pseudo_x, pseudo_y, pseudo_z])
    return p_pt, location


# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- Function that computes the LE & TE ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def segmenter(all_pts, all_edges, all_joints):

    # keeps a listing of all data points that are in the proper ordering for sectioning, TE then LE at each span
    final_pts_order = np.vstack((all_pts[6, :], all_pts[5, :]))

    # keeps a listing of all data points that have not been sectioned out yet
    available_pts = np.delete(all_pts, np.unique(np.where(all_pts == final_pts_order[0, :])[0]), axis=0)

    # initialize data to work with first loop
    order_y = [0, 1]
    test_pts = np.vstack((all_pts[0, :], all_pts[0, :]))
    edges = ["LE", "TE"]  # predefine the pt11 and pt12

    # -------------------------- Determine the sectioning of the current wing ------------------
    # loop 4 times to calculate the different wing sections
    for i in range(0, 4):
        # test to see if Pt 6 or 10 comes first
        test_pts = np.array([test_pts[order_y[1], :], all_pts[(i + 1), :]])
        order_x = np.argsort(test_pts[:, 0])
        order_y = np.argsort(test_pts[:, 1])

        # update the final points and the available points
        final_pts_order = np.vstack((final_pts_order, test_pts[order_y[0], :]))
        # tracks if the point is on the leading or trailing edge
        edges.append(all_edges[np.where(np.isin(available_pts, test_pts[order_y[0], :]))[0][0]])

    # Check how tucked the wing is - code will not work for wing shapes with Pt9 or Pt8 more interior than Pt10
    if all_pts[3, 1] < all_pts[1, 1] or all_pts[4, 1] < all_pts[1, 1]:
        position = "tucked"
    else:
        position = "untucked"

    # --------------------------- Create the sections ---------------------------------------------------

    # ------------------------- Section 1 -------------------------
    # this will work for both tucked and untucked wings

    # calculates the pseudo pt location and whether it is on the LE or TE
    pseudo_pt1, loc1 = det_pseudo_pt(final_pts_order[0:4, :], edges[3], edges[0:2].index("LE"), edges[0:2].index("TE"))
    segment1 = np.vstack((final_pts_order[0:3, :], pseudo_pt1))
    edges1 = edges[0:3]
    edges1.append(loc1)

    # Make segments 2 through 5
    if position == "tucked":
        le_pts = None
        te_pts = None
        airfoils = None
    else:
        # ------------------------- Section 2 -------------------------
        # adjust the input points if two leading edge points are specified after another
        if edges[3] == "LE" and edges[4] == "LE":  # if pt7 comes right after pt 6
            segment2_overshoot = np.vstack((segment1[2:4, :], final_pts_order[3, :], final_pts_order[5, :]))
            overshoot_edge = "TE"
        else:
            # take the last two points (pseudo and real) from the previous segment and add in the next two real points
            segment2_overshoot = np.vstack((segment1[2:4, :], final_pts_order[3:5, :]))
            overshoot_edge = "LE"

        # compute the pseudo point
        pseudo_pt2, loc2 = det_pseudo_pt(segment2_overshoot, overshoot_edge,
                                         edges1[2:4].index("LE"), edges1[2:4].index("TE"))
        # save final section data points and the LE or TE of the points
        segment2 = np.vstack((segment2_overshoot[0:3, :], pseudo_pt2))
        edges2 = edges1[2:4]
        edges2.append(edges[3])
        edges2.append(loc2)

        # ------------------------- Section 3 -------------------------
        segment3_overshoot = np.vstack((segment2[2:4, :], final_pts_order[4:6, :]))
        overshoot_edge = edges[5]

        # compute the pseudo point
        pseudo_pt3, loc3 = det_pseudo_pt(segment3_overshoot, overshoot_edge,
                                         edges2[2:4].index("LE"), edges2[2:4].index("TE"))
        # save final section data points and the LE or TE of the points
        segment3 = np.vstack((segment3_overshoot[0:3, :], pseudo_pt3))
        edges3 = edges2[2:4]
        edges3.append(edges[4])
        edges3.append(loc3)

        # ------------------------- Section 4 -------------------------
        segment4_overshoot = np.vstack((segment3[2:4, :], final_pts_order[5, :], all_pts[4, :]))
        # need to determine if the pseudo point goes on the top or the bottom based on if pt 7 is closer than pt 9
        if all_pts[2, 1] < all_pts[3, 1]:
            overshoot_edge = "LE"
        else:
            overshoot_edge = "TE"

        # compute the pseudo point
        pseudo_pt4, loc4 = det_pseudo_pt(segment4_overshoot, overshoot_edge,
                                         edges3[2:4].index("LE"), edges3[2:4].index("TE"))
        # save final section data points and the LE or TE of the points
        segment4 = np.vstack((segment4_overshoot[0:3, :], pseudo_pt4))
        edges4 = edges3[2:4]
        edges4.append(edges[5])
        edges4.append(loc4)

        # ------------------------- Section 5 -------------------------
        # this section will always be a triangle

        last_point = np.delete(available_pts, np.where(np.isin(available_pts, final_pts_order))[0], axis=0)
        # need to compute two pseudo points so that the twist of the final section can be determined
        # calculate at 80% distance to the last point from the previous section slice
        final_pseudo_y_pos = 0.8 * (last_point[0, 1] - segment4[2, 1]) + segment4[2, 1]
        segment5_overshoot = np.vstack((segment4[2:4, :], last_point, last_point))

        # compute the two pseudo point
        # need to specify the first row to have the array input properly to the function
        final_pseudo_le_x, final_pseudo_le_z = point_on_line(final_pseudo_y_pos, np.transpose(last_point[0, :]),
                                                             np.transpose(segment5_overshoot[edges4[2:4].index("LE")]))
        final_pseudo_te_x, final_pseudo_te_z = point_on_line(final_pseudo_y_pos, np.transpose(last_point[0, :]),
                                                             np.transpose(segment5_overshoot[edges4[2:4].index("TE")]))
        last_point_le = np.transpose(np.array([[final_pseudo_le_x], [final_pseudo_y_pos], [final_pseudo_le_z]]))
        last_point_te = np.transpose(np.array([[final_pseudo_te_x], [final_pseudo_y_pos], [final_pseudo_te_z]]))

        # save final section data points and the LE or TE of the points
        # only includes the previous slice
        segment5 = np.vstack((segment4[2:4, :]))
        edges5 = edges4[2:4]

        # ---------------------- Calculate the geometric properties of each section ------------------------------------

        le_pts = np.vstack((segment1[np.where(np.isin(edges1, "LE"))[0][0], :],
                            segment2[np.where(np.isin(edges2, "LE"))[0][0], :],
                            segment3[np.where(np.isin(edges3, "LE"))[0][0], :],
                            segment4[np.where(np.isin(edges4, "LE"))[0][0], :],
                            segment5[np.where(np.isin(edges5, "LE"))[0][0], :],
                            last_point))
        te_pts = np.vstack((segment1[np.where(np.isin(edges1, "TE"))[0][0], :],
                            segment2[np.where(np.isin(edges2, "TE"))[0][0], :],
                            segment3[np.where(np.isin(edges3, "TE"))[0][0], :],
                            segment4[np.where(np.isin(edges4, "TE"))[0][0], :],
                            segment5[np.where(np.isin(edges5, "TE"))[0][0], :],
                            last_point))
        # Defines what airfoil should be used for which segment bassed on the location of the main joints
        airfoils = ["InnerAirfoil"]*6
        for i in range(0, 6):
            if le_pts[i, 1] > all_joints[1, 1]:
                airfoils[i] = "MidAirfoil"
                if le_pts[i, 1] > all_joints[2, 1]:
                    airfoils[i] = "OuterAirfoil"

    return le_pts, te_pts, airfoils


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------- Function that computes the key geometry given the LE & TE --------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def geom_def(le_pts, te_pts):
    # initialize matrices
    full_dihedral = [0.0] * 6
    full_sweep = [0.0] * 6
    full_twist = [0.0] * 6
    full_chord = [0.0] * 6

    segment_span = [0.0] * 5
    true_span = [0.0] * 6
    full_span_frac = [0.0] * 6

    quarter_chord = np.zeros((6, 3))

    # ------- calculate the quarter-chord and total chord at each "slice" -------
    for i in range(0, 6):
        quarter_chord[i, :] = le_pts[i, :] + 0.25 * (te_pts[i, :] - le_pts[i, :])
        full_chord[i] = np.linalg.norm(te_pts[i, :] - le_pts[i, :])

    # ------- calculate the span without the dihedral to give the true span at each section -------
    # (only include the y and z) for c/4
    for i in range(0, 5):
        segment_span[i] = np.linalg.norm(quarter_chord[i+1, 1:3] - quarter_chord[i, 1:3])
        true_span[i+1] = segment_span[i] + true_span[i]

    # ------- calculate the span fraction at each "slice" -------
    for i in range(0, 6):
        full_span_frac[i] = true_span[i] / np.sum(segment_span)

    # -------------------------- Calculate the three major wing geometry angles --------------------------

    # initialize the first "slice" for twist
    full_twist[0] = 0.0

    # loop through the remaining "slices" - have 5 sections i.e. 6 slices
    for i in range(0, 5):
        # calculate wing dihedral of each section
        full_dihedral[i] = np.rad2deg(np.arctan(-(quarter_chord[i+1, 2]-quarter_chord[i, 2]) /
                                                 (quarter_chord[i+1, 1]-quarter_chord[i, 1])))
        # calculate wing sweep
        full_sweep[i] = np.rad2deg(np.arctan(-(quarter_chord[i+1, 0]-quarter_chord[i, 0]) /
                                             (segment_span[i])))
        # calculate wing twist
        if i < 4:
            full_twist[i+1] = np.rad2deg(np.arctan(-(quarter_chord[i+1, 2] - le_pts[i+1, 2]) /
                                                   (quarter_chord[i+1, 0] - le_pts[i+1, 0])))
        else:  # the twist at the tip is equal to the previous one
            full_twist[i + 1] = full_twist[i]

    return quarter_chord, full_chord, segment_span, full_span_frac, full_sweep, full_dihedral, full_twist

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------- Function that creates the airfoil dictionary ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def build_airfoil_dict(segments_re, airfoil_list):
    # initialize arrays
    airfoil_file = airfoil_list
    geometry_file = airfoil_list
    max_alpha = [0] * 7
    min_alpha = [0] * 7

    # create the file name for each airfoil at the applicable Re
    for i in range(0, 7):
        airfoil_file[i] = str(airfoil_list[i]).lower() + "_Re" + str(int(segments_re[i])) + ".0.txt"
        geometry_file[i] = str(airfoil_list[i]).lower() + ".txt"

        orig_dir = os.getcwd()
        new_dir = orig_dir + "\\airfoildat\\" + airfoil_list[i]
        os.chdir(new_dir)

        if not os.path.exists(airfoil_file[i]):
            wait = input("Create new airfoil file %s. Press enter to continue." % (airfoil_file[i]))
        # read in the airfoil file to save the maximum and minimum available alpha
        if i > 0:
            curr_airfoil = np.loadtxt(airfoil_file, skiprows= 1)
            max_alpha[i] = max(curr_airfoil[:,0])
            min_alpha[i] = min(curr_airfoil[:,0])

        os.chdir(orig_dir)

    # should have seven discrete airfoils (or airfoils at different Re) for each wing/body
    airfoil_dict = {}
    for i in range(0, 7):
        airfoil_dict[airfoil_list[i]] = {
            "type": "database",
            "input_file": airfoil_file[i],
            "outline_points": geometry_file[i]}

    return airfoil_dict, max_alpha, min_alpha

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------- Function that creates the airfoil dictionary ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def check_stalled_sections(final_alpha, final_sp_frac, max_alpha, min_alpha, full_span_frac):
    num_nodes = len(final_alpha)
    stall = [False]*num_nodes

    for i in range(0,num_nodes):
        if final_sp_frac[i] < full_span_frac[1]:
            if final_alpha[i] > max_alpha[0] | final_alpha[i] < min_alpha[0]:
                stall[i] = True
        if final_sp_frac[i] > full_span_frac[0] | final_sp_frac[i] < full_span_frac[2]:
            if final_alpha[i] > max_alpha[1] | final_alpha[i] < min_alpha[1]:
                stall[i] = True
        if final_sp_frac[i] > full_span_frac[1] | final_sp_frac[i] < full_span_frac[3]:
            if final_alpha[i] > max_alpha[2] | final_alpha[i] < min_alpha[2]:
                stall[i] = True
        if final_sp_frac[i] > full_span_frac[2] | final_sp_frac[i] < full_span_frac[4]:
            if final_alpha[i] > max_alpha[3] | final_alpha[i] < min_alpha[3]:
                stall[i] = True
        if final_sp_frac[i] > full_span_frac[3] | final_sp_frac[i] < full_span_frac[4]:
            if final_alpha[i] > max_alpha[4] | final_alpha[i] < min_alpha[4]:
                stall[i] = True

    return stall

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Function that defines the bird body --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def build_body_dict(body_root_airfoil, body_tip_airfoil, w_2, body_len, root_chord):
    y_body = np.linspace(0, w_2, 30)
    chord_body = body_len*np.cos((1/w_2)*math.acos(root_chord/body_len)*y_body)
    y_body = y_body/w_2
    body_dict = {
        "ID": 1,
        "side": "both",
        "is_main": True,
        "semispan": w_2,
        "twist": 0,
        "sweep": 0,
        "dihedral": 0,
        "chord": np.ndarray.tolist(np.vstack((y_body, chord_body)).T),
        "airfoil": [[0, body_root_airfoil], [1, body_tip_airfoil]],
        "grid": {"N": 50}}
    return body_dict


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ Function that creates the final wing dictionary ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def build_segment_wing_dict(bird_cg, bird_weight, segment_span, root_c4, full_ch, full_tw, full_sw, full_di,
                     body_dict, airfoil_list):
    wing_dict = {
        "CG": bird_cg,
        "weight": bird_weight,
        "reference": {},
        "controls": {},
        "airfoils": {
            "NACA0020": {
                "type": "linear",
                "aL0": 0.0,
                "CLa": 6.4336,
                "CmL0": 0.0,
                "Cma": 0.00,
                "CD0": 0.00513,
                "CD1": 0.0,
                "CD2": 0.0984,
                "CL_max": 1.4,
                "geometry": {
                    "NACA": "0022"}},
            "NACA0015": {
                "type": "linear",
                "aL0": 0.0,
                "CLa": 6.4336,
                "CmL0": 0.0,
                "Cma": 0.00,
                "CD0": 0.00513,
                "CD1": 0.0,
                "CD2": 0.0984,
                "CL_max": 1.4,
                "geometry": {
                    "NACA": "0015"}},
            "InnerAirfoil": {
                "type": "linear",
                "aL0": 0.0,
                "CLa": 6.4336,
                "CmL0": 0.0,
                "Cma": 0.00,
                "CD0": 0.00513,
                "CD1": 0.0,
                "CD2": 0.0984,
                "CL_max": 1.4,
                "geometry": {
                    "NACA": "0010"}},
            "MidAirfoil": {
                "type": "linear",
                "aL0": 0.0,
                "CLa": 6.4336,
                "CmL0": 0.0,
                "Cma": 0.00,
                "CD0": 0.00513,
                "CD1": 0.0,
                "CD2": 0.0984,
                "CL_max": 1.4,
                "geometry": {
                    "NACA": "0010"}},
            "OuterAirfoil": {
                "type": "linear",
                "aL0": 0.0,
                "CLa": 6.4336,
                "CmL0": 0.0,
                "Cma": 0.00,
                "CD0": 0.00513,
                "CD1": 0.0,
                "CD2": 0.0984,
                "CL_max": 1.4,
                "geometry": {
                    "NACA": "0010"}}},
        "wings": {
            "body": body_dict,
            "segment1": {
                "ID": 2,
                "side": "both",
                "is_main": True,
                "connect_to": {"ID": 1,
                               "dx": root_c4[0],
                               "dz": root_c4[2]},
                "semispan": segment_span[0],
                "twist": [[0, full_tw[0]], [1, full_tw[1]]],
                "sweep": full_sw[0],
                "dihedral": full_di[0],
                "chord": [[0, full_ch[0]], [1, full_ch[1]]],
                "airfoil": [[0, airfoil_list[0]], [1, airfoil_list[1]]],
                "grid": {"N": 50}},
            "segment2": {
                "ID": 3,
                "side": "both",
                "is_main": True,
                "connect_to": {"ID": 2},
                "semispan": segment_span[1],
                "twist": [[0, full_tw[1]], [1, full_tw[2]]],
                "sweep": full_sw[1],
                "dihedral": full_di[1],
                "chord": [[0, full_ch[1]], [1, full_ch[2]]],
                "airfoil": [[0, airfoil_list[1]], [1, airfoil_list[2]]],
                "grid": {"N": 50}},
            "segment3": {
                "ID": 4,
                "side": "both",
                "is_main": True,
                "connect_to": {"ID": 3},
                "semispan": segment_span[2],
                "twist": [[0, full_tw[2]], [1, full_tw[3]]],
                "sweep": full_sw[2],
                "dihedral": full_di[2],
                "chord": [[0, full_ch[2]], [1, full_ch[3]]],
                "airfoil": [[0, airfoil_list[2]], [1, airfoil_list[3]]],
                "grid": {"N": 50}},
            "segment4": {
                "ID": 5,
                "side": "both",
                "is_main": True,
                "connect_to": {"ID": 4},
                "semispan": segment_span[3],
                "twist": [[0, full_tw[3]], [1, full_tw[4]]],
                "sweep": full_sw[3],
                "dihedral": full_di[3],
                "chord": [[0, full_ch[3]], [1, full_ch[4]]],
                "airfoil": [[0, airfoil_list[3]], [1, airfoil_list[4]]],
                "grid": {"N": 50}},
            "segment5": {
                "ID": 6,
                "side": "both",
                "is_main": True,
                "connect_to": {"ID": 5},
                "semispan": segment_span[4],
                "twist": full_tw[4],
                "sweep": full_sw[4],
                "dihedral": full_di[4],
                "chord": [[0, full_ch[4]], [1, full_ch[5]]],
                "airfoil": [[0, airfoil_list[4]], [1, airfoil_list[5]]],
                "grid": {"N": 50}},
        }
    }

    return wing_dict


# ----------------------------------------------------------------------------------------------------------------------
# --------------------------- Function that creates the smoothed final wing dictionary ---------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def build_smooth_wing_dict(bird_cg, bird_weight, segment_span, root_c4,
                           full_span_frac, full_ch, full_tw, full_sw,
                           full_di, body_dict, airfoil_dict):

    wing_dict = {
        "CG": bird_cg,
        "weight": bird_weight,
        "reference": {},
        # "reference": {"area": total_area,
        #               "longitudinal_length": np.linalg.norm(root_c4)},
        #               NEED TO SET THIS UP TO NON-DIMENSIONALIZE BASED ON THE TOTAL WING AREA
        "controls": {},
        "airfoils": airfoil_dict,
        "wings": {
            "body": body_dict,
            "main_wing": {
                "ID": 2,
                "connect_to": {"ID": 1,
                               "dx": root_c4[0],
                               "dz": root_c4[2]},
                "side": "both",
                "is_main": True,
                "semispan": np.sum(segment_span),
                "twist": np.ndarray.tolist(np.transpose((np.vstack((full_span_frac, full_tw))))),
                # "twist": [[full_span_frac[0], full_tw[0]], [full_span_frac[1], full_tw[0]],
                #             [full_span_frac[1], full_tw[1]], [full_span_frac[2], full_tw[1]],
                #             [full_span_frac[2], full_tw[2]], [full_span_frac[3], full_tw[2]],
                #             [full_span_frac[3], full_tw[3]], [full_span_frac[4], full_tw[3]],
                #             [full_span_frac[4], full_tw[4]], [full_span_frac[5], full_tw[4]],
                #             [full_span_frac[5], full_tw[5]]],
                # "sweep": np.ndarray.tolist(np.transpose((np.vstack((full_span_frac, full_sweep))))),
                "sweep": [[full_span_frac[0], full_sw[0]], [full_span_frac[1], full_sw[0]],
                          [full_span_frac[1], full_sw[1]], [full_span_frac[2], full_sw[1]],
                          [full_span_frac[2], full_sw[2]], [full_span_frac[3], full_sw[2]],
                          [full_span_frac[3], full_sw[3]], [full_span_frac[4], full_sw[3]],
                          [full_span_frac[4], full_sw[4]], [full_span_frac[5], full_sw[4]],
                          [full_span_frac[5], full_sw[5]]],
                # "dihedral": np.ndarray.tolist(np.transpose((np.vstack((full_span_frac, full_di))))),
                "dihedral": [[full_span_frac[0], full_di[0]], [full_span_frac[1], full_di[0]],
                             [full_span_frac[1], full_di[1]], [full_span_frac[2], full_di[1]],
                             [full_span_frac[2], full_di[2]], [full_span_frac[3], full_di[2]],
                             [full_span_frac[3], full_di[3]], [full_span_frac[4], full_di[3]],
                             [full_span_frac[4], full_di[4]], [full_span_frac[5], full_di[4]],
                             [full_span_frac[5], full_di[5]]],
                "chord": np.ndarray.tolist(np.transpose((np.vstack((full_span_frac, full_ch))))),
                # "sweep": [[0.0, np.rad2deg(np.arctan(0.0125/(0.5*np.sum(segment_span))))], [0.5, np.rad2deg(np.arctan(0.0125/(0.5*np.sum(segment_span))))], [0.5, np.rad2deg(np.arctan(-0.0375/(0.5*np.sum(segment_span))))], [1.0, np.rad2deg(np.arctan(-0.0375/(0.5*np.sum(segment_span))))]],
                # "chord": [[0.0, 0.1],[0.5,0.05],[1.0,0.0]],
                #"dihedral": [[0.0, -10],[0.5,-10],[0.5,10],[1.0,10]],
                # "twist": [[0.0, 0],[0.3,0],[0.3,10],[0.6,10],[0.6,-10]],
                "airfoil": [[full_span_frac[0], airfoil_list[0]],
                            [full_span_frac[1], airfoil_list[1]],
                            [full_span_frac[2], airfoil_list[2]],
                            [full_span_frac[3], airfoil_list[3]],
                            [full_span_frac[4], airfoil_list[4]],
                            [full_span_frac[5], airfoil_list[5]]],
                "control_surface": {},
                "grid": {
                    "N": 250,
                    "cluster_points": np.ndarray.tolist(full_span_frac)}},
        }
    }

    return wing_dict
