"""
The purpose of this file is to read in the anatomical points defining a bird wing and to
output the appropriate dictionary to be used in MachUpX
"""
import numpy as np
import math
import os
import scipy

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
        order_y = np.argsort(test_pts[:, 1])

        # update the final points and the available points
        final_pts_order = np.vstack((final_pts_order, test_pts[order_y[0], :]))
        # tracks if the point is on the leading or trailing edge
        edges.append(all_edges[np.where(np.isin(available_pts, test_pts[order_y[0], :]))[0][0]])

    # Check how tucked the wing is - code will not work for wing shapes with:
    # 1/2. Pt9 or Pt8 more interior than Pt10
    # 3. Pt8 more interior than the wrist Pt4
    if all_pts[3, 1] < all_pts[1, 1] or all_pts[4, 1] < all_pts[1, 1] or all_pts[4, 1] < all_joints[1, 2]:
        position = "tucked"
    else:
        position = "untucked"

    # --------------------------- Create the sections ---------------------------------------------------

    # ------------------------- Section 1 ------------------------- # this will work for both tucked and untucked wings

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
        # save final section data points and the LE or TE of the points only includes the previous slice
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

        # Defines what airfoil should be used for which segment based on the location of the main joints
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


def geom_def(le_pts, te_pts, gam,body_gam, w_2):
    # initialize matrices
    discrete_dihedral = [0.0] * 5
    discrete_sweep = [0.0] * 5
    full_twist = [0.0] * 6
    full_chord = [0.0] * 6

    dis_segment_span = [0.0] * 5
    dis_true_span = [0.0] * 6

    quarter_chord = np.zeros((6, 3))

    # ------- calculate the quarter-chord and total chord at each "slice" -------
    for i in range(0, 6):
        quarter_chord[i, :] = le_pts[i, :] + 0.25 * (te_pts[i, :] - le_pts[i, :])
        full_chord[i] = np.linalg.norm(te_pts[i, :] - le_pts[i, :])

    # ------- calculate the span without the dihedral to give the true span at each section -------
    # (only include the y and z) for c/4
    for i in range(0, 5):
        dis_segment_span[i] = np.linalg.norm(quarter_chord[i+1, 1:3] - quarter_chord[i, 1:3])
        dis_true_span[i+1] = dis_segment_span[i] + dis_true_span[i]
    dis_span_frac = dis_true_span / np.sum(dis_segment_span)
    # -------------------------- Calculate the three major wing geometry angles --------------------------

    # initialize the first "slice" for twist
    full_twist[0] = 0.0

    # loop through the remaining "slices" - have 5 sections i.e. 6 slices
    for i in range(0, 5):
        # calculate wing dihedral of each section, will use to inform initial guess in radians
        discrete_dihedral[i] = np.arctan(-(quarter_chord[i+1, 2]-quarter_chord[i, 2]) /
                                                    (quarter_chord[i+1, 1]-quarter_chord[i, 1]))
        # calculate wing sweep, will use to inform initial guess
        discrete_sweep[i] = np.arctan(-(quarter_chord[i+1, 0]-quarter_chord[i, 0]) /
                                                 (dis_segment_span[i]))
        # calculate wing twist
        if i < 4:
            full_twist[i+1] = np.rad2deg(np.arctan(-(quarter_chord[i+1, 2] - le_pts[i+1, 2]) /
                                                   (quarter_chord[i+1, 0] - le_pts[i+1, 0])))
        else:  # the twist at the tip is equal to the previous one
            full_twist[i + 1] = full_twist[i]

    # ------------------------ Blend the distributions ------------------------------

    # ------- calculate the DIHEDRAL blend of each segment -------
    # input must be in radians
    dis_body_di = np.arctan(-quarter_chord[0, 2] / quarter_chord[0, 1])
    # -- Initial guess --
    initial_guess_di = discrete_dihedral
    initial_guess_di.insert(0, dis_body_di)  # already in radians
    initial_guess_di = np.array(initial_guess_di) - np.deg2rad(2)  # subtract 2 degrees to undershoot
    # solve for dihedral
    sol_di = scipy.optimize.fsolve(solve_blend, initial_guess_di,
                                   args=(quarter_chord[:, 1], quarter_chord[:, 2], gam, body_gam, w_2))
    dihedral = np.rad2deg(sol_di)
    test_di = solve_blend(sol_di, quarter_chord[:, 1], quarter_chord[:, 2], gam, body_gam, w_2)
    # ------- calculate the true span of each segment -------
    # input must be in radians
    segment_span, true_span, body_seg_span, body_span = calc_true_span(sol_di, quarter_chord[:, 1], gam, body_gam, w_2)
    full_span_frac = true_span / np.sum(segment_span)
    full_body_frac = body_span / np.sum(body_seg_span)
    true_body_w2 = np.sum(body_seg_span)
    # ------- calculate the SWEEP blend of each segment -------
    # the y, gamma and length inputs must be the newly computed true span
    rcy_sw = np.array([true_span[0], true_span[2], true_span[4], true_span[6], true_span[8], true_span[9]]) + true_body_w2
    gam_sw = np.array([(true_span[1]/true_span[2]), (true_span[3]/true_span[4]),
                       (true_span[5]/true_span[6]), (true_span[7]/true_span[8])])
    # input must be in radians
    dis_body_sw = np.arctan(-quarter_chord[0, 0] / true_body_w2)
    # -- Initial guess --
    initial_guess_sw = discrete_sweep
    initial_guess_sw.insert(0, dis_body_sw)  # already in radians
    initial_guess_sw = np.array(initial_guess_sw) - np.deg2rad(2)
    # solve for sweep
    sol_sw = scipy.optimize.fsolve(solve_blend, initial_guess_sw,
                                   args=(rcy_sw, quarter_chord[:, 0], gam, full_body_frac[1:5], true_body_w2))
    sweep = np.rad2deg(sol_sw)
    test_sw = solve_blend(sol_sw, rcy_sw, quarter_chord[:, 0], gam, full_body_frac[1:5], true_body_w2)

    # ------------------------ Save the distributions ------------------------------
    # organize the dihedral angles to line up with the span fractions used
    wing_dihedral = []
    wing_sweep = []

    for i in range(1, 6):
        wing_dihedral = wing_dihedral + [dihedral[i], dihedral[i]]
        wing_sweep = wing_sweep + [sweep[i], sweep[i]]

    body_sweep = sweep[0]
    body_dihedral = dihedral[0]

    return quarter_chord, full_chord, segment_span, full_span_frac, dis_span_frac,\
           wing_sweep, wing_dihedral, full_twist, full_body_frac, true_body_w2, body_sweep, body_dihedral

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------- Function that solves for dihedral and sweep blend ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def solve_blend(x, rcy, rcz_x, wing_gam, body_gam, body_w):
    # incoming dihedral angles must be in radians!
    newcon = x
    segz_x = np.zeros(6)
    fun = np.empty(6)

    # --------------------- Body ---------------------
    # ----- Segment 1 ------
    gam0 = body_gam[0]
    gam1 = body_gam[1]
    gam2 = body_gam[2]

    seg_len = body_w  # along y
    # First linear region
    a1 = newcon[0] / (gam1 - gam0)  # boundary condition on the first linear slope start
    b1 = - a1 * gam0  # boundary condition on the linear slope end di = a*y + b - this is in the 0 to 1 y axis
    # Second linear region
    a2 = (newcon[1] - newcon[0]) / (1 - gam2)  # boundary condition on the linear slope start
    b2 = newcon[1] - a2  # boundary condition on the linear slope end di = a*y + b - this is in the 0 to 1 y axis
    # Note the negatives are necessary because a +ve sweep gives -ve x
    segz_x_1 = - (scipy.integrate.fixed_quad(lambda y: np.tan(a1 * y + b1), gam0, gam1, n=12))[0] * seg_len
    segz_x_2 = - np.tan(newcon[0]) * (gam2 - gam1) * seg_len
    segz_x_3 = - (scipy.integrate.fixed_quad(lambda y: np.tan(a2 * y + b2), gam2, 1, n=12))[0] * seg_len
    fun[0] = segz_x_1 + segz_x_2 + segz_x_3 - rcz_x[0]

    # --------------------- Wing ---------------------
    # ----- Segment 1 ------
    for i in range(1, 5):
        seg_len = (rcy[i] - rcy[i-1])
        a = (newcon[i+1] - newcon[i])/(1-wing_gam)  # boundary condition on the linear slope start
        b = newcon[i+1] - a  # boundary condition on the linear slope end di = a*y + b - this is in the 0 to 1 y axis

        segz_x[i] = -np.tan(newcon[i])*wing_gam*seg_len - (scipy.integrate.fixed_quad(lambda y: np.tan(a*y+b), wing_gam, 1, n=10))[0]*seg_len
        fun[i] = segz_x[i] - (rcz_x[i]-rcz_x[i-1])

    # ----- Segment 5 ------
    # last segment is not blended
    fun[5] = -np.tan(newcon[5])*(rcy[5]-rcy[4]) - (rcz_x[5]-rcz_x[4])

    return fun

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------- Function that computes the true wing span with input dihedral --------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def calc_true_span(di, rcy, gam, body_gam, w_2):

    # ------------------------------- Body -------------------------------------
    gam0 = body_gam[0]
    gam1 = body_gam[1]
    gam2 = body_gam[2]

    true_body_full_span = np.zeros(5)
    true_body_seg_len = np.empty(4)

    seg_len = w_2
    # First linear region
    a1 = di[0] / (gam1-gam0)  # boundary condition on the first linear slope start
    b1 = - a1*gam0  # boundary condition on the linear slope end di = a*y + b - this is in the 0 to 1 y axis
    # Second linear region
    a2 = (di[1] - di[0]) / (1 - gam2)  # boundary condition on the linear slope start
    b2 = di[1] - a2  # boundary condition on the linear slope end di = a*y + b - this is in the 0 to 1 y axis

    true_body_seg_len[0] = gam0*seg_len
    true_body_seg_len[1] = (scipy.integrate.fixed_quad(lambda y: 1 / np.cos(a1 * y + b1), gam0, gam1, n=12))[0]*seg_len
    true_body_seg_len[2] = seg_len*(gam2-gam1)/(np.cos(di[0]))
    true_body_seg_len[3] = (scipy.integrate.fixed_quad(lambda y: 1 / np.cos(a2 * y + b2), gam2, 1, n=12))[0] * seg_len

    for i in range(1, 5):
        true_body_full_span[i] = np.sum(true_body_seg_len[0:i])

    # ------------------------------- Wing -------------------------------------
    true_full_span = np.zeros(10)
    true_seg_len = np.zeros(9)
    count = 0
    for i in range(0, 4):
        seg_len = (rcy[i + 1] - rcy[i])
        a = (di[i+2] - di[i+1]) / (1 - gam)  # boundary condition on the linear slope start
        b = di[i+2] - a  # boundary condition on the linear slope end di = a*y + b - this is in the 0 to 1 y axis
        true_seg_len[count] = seg_len*gam/(np.cos(di[i+1]))
        true_seg_len[count+1] = (scipy.integrate.fixed_quad(lambda y: 1 / np.cos(a*y+b), gam, 1, n=12))[0]*seg_len
        count += 2
    # last segment is not blended
    true_seg_len[8] = (rcy[5]-rcy[4]) / (np.cos(di[5]))

    for i in range(1, 10):
        true_full_span[i] = np.sum(true_seg_len[0:i])

    return true_seg_len, true_full_span, true_body_seg_len, true_body_full_span

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------- Function that creates the airfoil dictionary ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def build_airfoil_dict(segments_re, airfoil_list):
    # initialize arrays
    abs_geom_path = ['0'] * len(segments_re)
    abs_file_path = ['0']*len(segments_re)
    airfoil_name = ['0'] *len(segments_re)
    geometry_file = ['0']*len(segments_re)
    check_NACA = [False]*len(segments_re)

    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

    # create the file name for each airfoil at the applicable Re
    for i in range(0, len(segments_re)):
        # need to add the absolute path to each of these airfoils
        # ensure that the airfoils are saved within airfoildat
        # - Data file -
        airfoil_file = str(airfoil_list[i]).lower() + "_Re" + str(int(segments_re[i])) + ".txt"
        rel_dat_path = "airfoildat\\" + str(airfoil_list[i]).lower() + "\\" + str(airfoil_file)
        abs_file_path[i] = os.path.join(script_dir, rel_dat_path)
        # - Geometry file -
        geometry_file = str(airfoil_list[i]).lower() + "_geometry.csv"
        rel_geom_path = "airfoildat\\" + str(airfoil_list[i]).lower() + "\\" + str(geometry_file)
        abs_geom_path[i] = os.path.join(script_dir, rel_geom_path)
        # - Save the airfoil name -
        airfoil_name[i] = str(airfoil_list[i]).lower() + "_Re" + str(int(segments_re[i]))

        check_NACA[i] = "naca" in str(airfoil_list[i]).lower()

    # should have eight discrete airfoils (or airfoils at different Re) for each wing/body
    airfoil_dict = {}
    for i in range(0, len(segments_re)):
        if check_NACA[i]:
            naca_name = airfoil_list[i]  # save name of airfoil for later
            naca_num = naca_name[4:]
            airfoil_dict[airfoil_name[i]] = {
                "type": "database",
                "input_file": abs_file_path[i],
                "geometry": {"NACA": naca_num}}
        else:
            airfoil_dict[airfoil_name[i]] = {
                "type": "database",
                "input_file": abs_file_path[i],
                "geometry": {"outline_points": abs_geom_path[i]}}

    return airfoil_dict


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Function that defines the bird body --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def build_body_dict(airfoils, true_w_2, body_len, root_chord, body_span_frac, di_bodycon, di_root, sw_bodycon, sw_root):
    # the distributions of sweep and dihedral must match those at the root as well as the total displacement of the
    # wing root quarter chord from 0 this requires a minimization process to solve because there is no direct
    # analytical solution

    # calculate the chord distribution as a cosine function
    y_body = np.linspace(0, true_w_2, 30)
    chord_body = body_len*np.cos((1/true_w_2)*math.acos(root_chord/body_len)*y_body)
    y_body = y_body/true_w_2  # makes into the span fraction of the body

    body_dict = {
        "ID": 1,
        "side": "both",
        "is_main": True,
        "semispan": true_w_2,
        "twist": 0,
        "sweep": [[0, 0], [body_span_frac[1], 0],
                  [body_span_frac[2], sw_bodycon], [body_span_frac[3], sw_bodycon], [1, sw_root]],
        "dihedral": [[0, 0], [body_span_frac[1], 0],
                     [body_span_frac[2], di_bodycon], [body_span_frac[3], di_bodycon], [1, di_root]],
        "chord": np.ndarray.tolist(np.vstack((y_body, chord_body)).T),
        "airfoil": [[0, airfoils[0]], [body_span_frac[1], airfoils[1]], [1, airfoils[2]]],
        "grid": {"N": 100}}

    return body_dict


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ Function that creates the final wing dictionary ---------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def build_segment_wing_dict(bird_cg, bird_weight, segment_span, full_ch, full_tw, full_sw, full_di,
                            body_dict, airfoil_dict, airfoil_list):
    wing_dict = {
        "CG": bird_cg,
        "weight": bird_weight,
        "reference": {},
        "controls": {},
        "airfoils": airfoil_dict,
        "wings": {
            "body": body_dict,
            "segment1": {
                "ID": 2,
                "side": "both",
                "is_main": True,
                "connect_to": {"ID": 1},
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


def build_smooth_wing_dict(bird_cg, bird_weight, segment_span, full_span_frac, full_ch, full_tw, full_sw,
                           full_di, dis_span_frac, body_dict, airfoil_dict, airfoil_list):

    wing_dict = {
        "CG": bird_cg,
        "weight": bird_weight,
        "reference": {},
        "controls": {},
        "airfoils": airfoil_dict,
        "wings": {
            "body": body_dict,
            "main_wing": {
                "ID": 2,
                "connect_to": {"ID": 1},
                "side": "both",
                "is_main": True,
                "semispan": np.sum(segment_span),
                "twist": np.ndarray.tolist(np.transpose((np.vstack((dis_span_frac, full_tw))))),
                "sweep": np.ndarray.tolist(np.transpose((np.vstack((full_span_frac, full_sw))))),
                "dihedral": np.ndarray.tolist(np.transpose((np.vstack((full_span_frac, full_di))))),
                "chord": np.ndarray.tolist(np.transpose((np.vstack((dis_span_frac, full_ch))))),
                "airfoil": [[dis_span_frac[0], airfoil_list[2]],
                            [dis_span_frac[1], airfoil_list[3]],
                            [dis_span_frac[2], airfoil_list[4]],
                            [dis_span_frac[3], airfoil_list[5]],
                            [dis_span_frac[4], airfoil_list[6]],
                            [dis_span_frac[5], airfoil_list[7]]],
                "control_surface": {},
                "grid": {
                    "N": 300,
                    "cluster_points": list(dis_span_frac)}},
        }
    }

    return wing_dict

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------- Function that creates the smoothed final wing dictionary ---------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def check_build_error(quarter_chord, cpx, cpy, cpz):
    error_c4 = [0.0]*6
    error_c4[0] = np.sqrt((quarter_chord[0][0] - cpx[0])**2 +
                          (quarter_chord[0][1] - cpy[0])**2 +
                          (quarter_chord[0][2] - cpz[0])**2)
    error_c4[5] = np.sqrt((quarter_chord[5][0] - cpx[len(cpx)-1])**2 +
                          (quarter_chord[5][1] - cpy[len(cpx)-1])**2 +
                          (quarter_chord[5][2] - cpz[len(cpx)-1])**2)

    for j in range(1, len(quarter_chord)-1):
        error_c4[j] = min(np.sqrt((quarter_chord[j][0] - cpx)**2 +
                                  (quarter_chord[j][1] - cpy)**2 +
                                  (quarter_chord[j][2] - cpz)**2))

    return error_c4