#  PyME code to compute the SIR
#  Author: Vikram Sunkara

import numpy as np


#import sys
#sys.path.append('/Users/sunkara/dev/PyME/') 

import pyme

from pyme.model import Model 

##### model class #####

"""
This is a CMEPY model file. We simple need to give propensity functions, transition vectors, the initial state and the species names. 

"""

SIR_model = Model(	propensities = [ lambda *x : 0.3*np.maximum(x[0],0.0)*np.maximum(x[1],0.0),
											lambda *x : 1.5*np.maximum(x[1],0.0)],
								transitions =  [(-1,1),
												(0,-1)],
								#shape = (2,2),
								initial_state = (200,4),
								species = ("S","I"))


delta_t = 0.001
T = np.arange(0.01,0.1,delta_t)


### OFSP Compuation

from pyme.OFSP import OFSP_Solver

"""
	def __init__(self,model,compress_window,step_error,expander_name="SE1",validity_test=None):

		OFSP solver for solving the CME for a given model over time.

		Parameters
		------------------------
		model 			: CMEPY Model Class
		compress_window : int , 
							number of steps before compressing the domain.
		step_error		: float, 
							global error allowed in the sink state per step.
		expander_name 	: str , 
							"SE1" Simple N-step expander.
		validity_test	: func , 
							Validity function is by default looking for non negative states
"""

OFSP_obj = OFSP_Solver(SIR_model,50,1e-6,expander_name="SE1")

for t in T:
	OFSP_obj.step(t)
	OFSP_obj.plot(inter=True)			# Interactive mode graphs the marginal each time called.
	OFSP_obj.print_stats
	OFSP_obj.check_point()
	X = np.array([[150,180],[50,20]])	# We can probe various states for their probabilities.
	OFSP_obj.probe_states(X)			# Probes and stores the states of the respective time step.

OFSP_obj.plot_checked()


from pyme.Hybrid_FSP import Hybrid_FSP_solver
"""
Initialising a Hybrid solver class. Where we want to consider S to be stochastic and I to be deterministic

	def __init__(self,model,stoc_vector,model_name,sink_error,jac=None): 

		@param model 		: model.Model. 
		@param stoc_vector 	: numpy.ndarray.  
		@param model_name 	: str, 'MRCE' or 'HL'.
		@param sink_error 	: float, maximum error allowed in the sink state.
		@param jac 			: (List of Functions), The jacobian of the propensity functions.


"""

stoc_vector = np.array([True,False])

# Evolving the density a bit forward so that we have a regular marginal.
OFSP_obj = OFSP_Solver(SIR_model,"SE1",50,1e-6)
OFSP_obj.step(0.01)

### Hybrid MRCE Computation
MRCE_obj = Hybrid_FSP_solver(SIR_model,stoc_vector,"MRCE",1e-7)
MRCE_obj.set_initial_values(OFSP_obj.domain_states,OFSP_obj.p,t=0.01)

for t in T:
	MRCE_obj.step(t)
	MRCE_obj.print_stats
	#MRCE_obj.plot(inter=True)
	MRCE_obj.check_point()
	X = np.array([[150.0,180.0],[1,2]])
	MRCE_obj.probe_states(X)

MRCE_obj.plot_checked()


### Hybrid HL Computation
from pyme.Hybrid_FSP import Hybrid_FSP_solver
stoc_vector = np.array([True,False])

HL_obj = Hybrid_FSP_solver(SIR_model,stoc_vector,"HL",1e-7)
HL_obj.set_initial_values(OFSP_obj.domain_states,OFSP_obj.p,t=0.01)
for t in T:
	HL_obj.step(t)
	HL_obj.print_stats
	HL_obj.check_point()
	X = np.array([[150.0,180.0],[1,2]])
	HL_obj.probe_states(X)

HL_obj.plot_checked()

