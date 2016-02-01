"""
First OFSP solver using the ABC model. 
Code for the tutorial http://github.com/vikramsunkara/PyME/wiki/OFSP-Solver-Tutorial
"""

import numpy as np

# Importing the model class
from ABC_model import ABC_model

# calling the class
from pyme.OFSP import OFSP_Solver

# OFSP solver for the ABC model
OFSP_ABC = OFSP_Solver(ABC_model,10,1e-6)

# Example with Validity function
"""
def validity_func(X):
  return np.sum(np.abs(X),axis=0) == 10 # since we started with 10 states.
# The solver initialisation would look like

OFSP_ABC = OFSP_Solver(ABC_model,10,1e-6,validity_test = validity_func)

"""


T = np.arange(0.01,1.0,0.01)
for t in T:
	OFSP_ABC.step(t)
	OFSP_ABC.print_stats 		# Prints some information of where the solver is.

	""" Runtime plotting"""
	#OFSP_ABC.plot(inter=True)  # For interactive plotting

	""" Check Point"""
	OFSP_ABC.check_point()

	""" Probing """
	X = np.zeros((3,2))
	X[:,0] = [8,2,0]
	X[:,1] = [7,2,1]
	OFSP_ABC.probe_states(X)


OFSP_ABC.plot()
OFSP_ABC.plot_checked()