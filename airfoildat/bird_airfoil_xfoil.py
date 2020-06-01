"""
The purpose of this file is to read in the selected avian airfoils and run XFOIL across the specified Re
Still requires a good amount of manual intervention
"""
import airfoil_db as adb
import math as m

# read in the current airfoil .txt file - set up so top comes first
geometry_file = "lius20.txt"
airfoil_name, type_of_file = geometry_file.split(".") # save name of airfoil for later

airfoil_input = {
    "type": "database",
    "geometry": {
        "outline_points": geometry_file,
        "top_first": True
        # "NACA": "3603"
        }
}

# define the current airfoil class
airfoil = adb.Airfoil(airfoil_name, airfoil_input, verbose=False)

# --- Define the test ranges for the airfoil
alpha_range = [m.radians(-20), m.radians(0)]  # set so alpha varies from -10 to 10 with 0.25deg
alpha_steps = int((m.degrees(alpha_range[1]) - m.degrees(alpha_range[0]))*20 + 1)

re_range = [400000, 400000]  # set so re varies with 5E4deg
re_steps = int((re_range[1] - re_range[0])/50000 + 1)

dof = {
    "alpha": {"range": alpha_range, "steps": alpha_steps, "index": 1},
    "Rey": {"range": re_range, "steps": re_steps, "index": 2},
    "Mach": 0.04}

# Generate or import database
airfoil.generate_database(degrees_of_freedom=dof, N=300, max_iter=1000, show_xfoil_output=True, verbose=False)
