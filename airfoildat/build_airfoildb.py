"""
The purpose of this file is to read in the pacc files and build up the airfoil database for use in birdwingsegmenter
"""

import airfoil_db as adb
import os
import numpy as np

geometry_file = "lius40_geometry.csv"
# save name of airfoil for later
airfoil_name, type_of_file = geometry_file.split("_")
orig_dir = os.getcwd()
new_dir = orig_dir + "\\" + airfoil_name
os.chdir(new_dir)

# Create header
header = []
# Add coefficients
header.append("{:<25s}".format('#alpha'))
header.append("{:<25s}".format('CL'))
header.append("{:<25s}".format('CD'))
header.append("{:<25s}".format('Cm'))
header = " ".join(header)

airfoil_input = {
    "type": "database",
    "geometry": {
        "outline_points": geometry_file,
        "top_first": True
        #"NACA": "0020"
    }
}
# define the current airfoil class
airfoil = adb.Airfoil(airfoil_name, airfoil_input, verbose=False)
camber = airfoil.get_outline_points()
quarter_chord_camber = airfoil.get_camber(0.25)
# go into the folder with the airfoil name and get all filenames
for root, dirs, files in os.walk("."):
    for filename in files:
        if ".pacc" in filename:
            # read in data from the pacc file
            alpha, Cl, Cd, Cm, Re, M = airfoil.read_pacc_file(filename)
            u, unique_indices = np.unique(alpha, return_index=True)
            # define new filename for the database
            new_filename = airfoil_name + "_Re" + str(int(Re)) + ".txt"
            # paste each column together
            full_dat = np.transpose((np.vstack((alpha[unique_indices], Cl[unique_indices],
                                                Cd[unique_indices], Cm[unique_indices]))))
            # sort by angle of attack
            full_dat = full_dat[full_dat[:, 0].argsort()]

            with open(new_filename, 'wb') as f:
                np.savetxt(f, full_dat, fmt='%.5f', header=header, comments='')

os.chdir(orig_dir)
