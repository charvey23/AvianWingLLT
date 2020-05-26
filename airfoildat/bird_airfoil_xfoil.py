"""
The purpose of this file is to read in the selected avian airfoils and to create the appropriate XFoil data base files
across the specified Re
"""
import airfoil_db as adb
import math as m

# read in the current airfoil .txt file - set up so top comes first
geometry_file = "e420.txt"
# save name of airfoil for later
airfoil_name, type_of_file = geometry_file.split(".")

airfoil_input = {
    "type": "database",
    "geometry": {
        "outline_points": geometry_file,
        "top_first": True
        # "NACA" : "9412"
    }
}
# define the current airfoil class
airfoil = adb.Airfoil("E420", airfoil_input, verbose=False)
#set so alpha vaires from -20 to 20 with 0.5deg
alpha_range = [m.radians(-20.0), m.radians(20.0)]
alpha_steps = 81
# define the two test ranges for the airfoil
re_range_low = [10000, 100000]
re_steps_low = 10
re_range_hi = [150000, 800000]
re_steps_hi = 8

dof_low = {
    "alpha": {"range": alpha_range, "steps": alpha_steps, "index": 1},
    "Rey": {"range": re_range_low, "steps": re_steps_low, "index": 2},
    "Mach": 0.03
    }

dof_hi = {
    "alpha": {"range": alpha_range, "steps": alpha_steps, "index": 1},
    "Rey": {"range": re_range_hi, "steps": re_steps_hi, "index": 2},
    "Mach": 0.03
    }

# Generate or import database
airfoil.generate_database(degrees_of_freedom=dof_low, max_iter=10000, show_xfoil_output=False)
curr_filename = airfoil_name + "_low.txt"
airfoil.export_database(filename=curr_filename)

airfoil.generate_database(degrees_of_freedom=dof_hi, max_iter=10000, show_xfoil_output=False)
curr_filename = airfoil_name + "_hi.txt"
airfoil.export_database(filename=curr_filename)