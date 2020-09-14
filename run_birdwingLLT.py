import machupX as MX
import json
import os
import run_birdwingLLT as rbw
import airfoil_db as adb

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------- Function that creates the smoothed final wing dictionary ---------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def run_machupx(curr_wing_name, test_v, test_aoa, density, kin_vis, curr_wing_dict, max_it, tol_con, relax):

    input_file = rbw.create_inputfile(density, kin_vis, curr_wing_name, curr_wing_dict, test_v,
                                      test_aoa, max_it, tol_con, relax)

    # get relative path for the loads file
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    results_file = curr_wing_name + "_U" + str(test_v) + "_alpha" + str(test_aoa) + "_results.json"
    abs_res_path = os.path.join(script_dir, "DataConverged\\" + str(results_file))
    abs_input_path = os.path.join(script_dir, "InputFiles\\" + curr_wing_name + 'inputs.json')

    # save the current input file once per wing
    if test_aoa == -5:
        with open(abs_input_path, 'w') as outfile:
            json.dump(input_file, outfile)

    try:
        # Initialize Scene object.
        my_scene = MX.Scene(input_file)
        # set the errors to raise when iterating
        my_scene.set_err_state(not_converged="raise", database_bounds="raise")

        # save the final reference geometry used in the analysis
        final_geom = my_scene.get_aircraft_reference_geometry()
        mac_curr = my_scene.MAC()  # save the mean aerodynamic chord

        # solve and save the loads
        results = my_scene.solve_forces(dimensional=True, non_dimensional=True,
                                        verbose=True, report_by_segment=True, filename=abs_res_path)

        # save the distributions if no error was thrown above
        dist_file = curr_wing_name + "_U" + str(test_v) + "_alpha" + str(test_aoa) + "_dist.json"
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
        final_geom = None
        mac_curr = None
        results = None
        curr_dist = None

    return converged, err, final_geom, mac_curr, results, curr_dist

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------- Function that creates the input file into MachUpX ---------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def create_inputfile(density, kin_vis, curr_wing_name, curr_wing_dict, test_v, test_aoa, max_it, tol_con, relax):
    input_file = {
        "tag": "Aerodynamic properties of simplified bird wing",
        "run": {
            "display_wireframe": {"show_legend": True},
            "solve_forces": {"non_dimensional": True},
            "distributions": {}},
        "solver": {
            "type": "nonlinear",
            "relaxation": relax,
            "max_iterations": max_it,
            'convergence': tol_con
        },
        "units": "SI",
        "scene": {
            "atmosphere": {
                "rho": density,
                "viscosity": kin_vis
            },
            "aircraft": {
                curr_wing_name: {
                    "file": curr_wing_dict,
                    "state": {
                        "type": "aerodynamic",
                        "velocity": test_v,
                        "alpha": test_aoa}, }}}
    }
    return input_file
