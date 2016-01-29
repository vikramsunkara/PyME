"""
Simple Viral Dynamics model. 

Species
--------------------
T = Templetes
G = Genome
S = Structures
V = Virons

Reactions
--------------------

r1 : T -> T + G
r2 : G -> T
r3 : T -> T + S
r4 : T -> *
r5 : S -> *
r6 S + G -> V

"""

import numpy as np

# Set Path
import sys
sys.path.append('/Users/sunkara/dev/PyME/') 

from model import Model


##### CMEPY Model class

Viral_D_model = Model(	propensities = [ lambda *x : 1.0*np.maximum(x[0],0.0),
									lambda *x : 0.025*np.maximum(x[1],0.0),
									lambda *x : 1000.0*np.maximum(x[0],0.0),
									lambda *x : 0.25*np.maximum(x[0],0.0),
									lambda *x : 2.0*np.maximum(x[2],0.0),
									lambda *x : 7.5e-6*np.maximum(x[2],0.0)*np.maximum(x[1],0.0)],
								transitions =  [(0,1,0,0),
												(1,-1,0,0),
												(0,0,1,0),
												(-1,0,0,0),
												(0,0,-1,0),
												(0,-1,-1,1)],
								shape = (2,2,2,2),
								initial_state = (6,0,0,0),
								species = ("T","G","S","V")
								)

def jac_0(x):
	return np.array([0.0,0.0,0.0,0.0])
def jac_1(x):
	return np.array([0.0,0.025*x[1],0.0,0.0])
def jac_5(x):
	return np.array([0.0,0.0,2.0*x[2],0.0])
def jac_6(x):
	return np.array([0.0,7.5e-6*x[1]*x[2],7.5e-6*x[1]*x[2],0.0])

def Jac_Mat(X,deter_vec,i):
    jac_list = [jac_0,jac_1,jac_0,jac_0,jac_5,jac_6]
    mat = np.array(map(jac_list[i],X.T)).T
    return mat[deter_vec,:]

delta_t = 0.1
T = np.arange(0.005,20.0,delta_t)


### Using the OFSP to generate a regular initial condition.
from OFSP import OFSP_Solver

OFSP_obj = OFSP_Solver(Viral_D_model,"SE1",5,1e-6)
OFSP_obj.step(0.005)
OFSP_obj.print_stats

####### Hybrid Solver Class ########

from Hybrid_FSP import Hybrid_FSP_solver

### Hybrid MRCE Computation
stoc_vector = np.array([True,False,False,True])
"""
Templetes and Virons are stochastic.
Genome and Structures are modelled by conditional / marginal expectations.
"""


MRCE_obj = Hybrid_FSP_solver(Viral_D_model,stoc_vector,"MRCE",1e-7,jac=Jac_Mat)
MRCE_obj.set_initial_values(OFSP_obj.domain_states,OFSP_obj.p,t=0.005)

for t in T:
	MRCE_obj.step(t)
	MRCE_obj.print_stats
	MRCE_obj.plot(inter=True)

### Hybrid HL Computation

HL_obj = Hybrid_FSP_solver(Viral_D_model,stoc_vector,"HL",1e-7)
HL_obj.set_initial_values(OFSP_obj.domain_states,OFSP_obj.p,t=0.005)
for t in T:
	HL_obj.step(t)
	HL_obj.print_stats


