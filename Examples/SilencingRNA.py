"""
Silencing RNA model 
Author:Vikram Sunkara


Species
---------------------
Gene B : poisiton B in the genome
sRNA  : Silencing RNA
mRNA   : mRNA produced by Gene A (Always On)
A      : Protein A made by mRNA

Reactions
----------------------

r1 : * -> mRNA
r2 : mRNA -> A
r3 : Gene B -> sRNA + Gene B
r4 : sRNA + mRNA -> *
r5 : A + Gene B -> A
r6 : * -> Gene B

"""

import numpy as np

# Set Path
#import sys
#sys.path.append('/Users/sunkara/dev/PyME/') 

import pyme

from pyme.model import Model 

sRNA_model= Model(
    propensities = [lambda *x : 0.1,
                    lambda *x : 0.05*np.maximum(x[2],0.0),
                    lambda *x : 0.02*np.maximum(x[2],0.0)*np.maximum(x[1],0.0),
                    lambda *x : 0.1*np.maximum(x[0],0.0), # 0.1 First parameter
                    lambda *x : 0.001*np.maximum(x[3],0.0)*np.maximum(x[0],0.0), # 0.001 Second parameter
                    lambda *x : 0.01*np.where(x[0] == 0,1.0,0.0) #np.maximum(1 - x[0],0.0)
                    ],
    transitions = [(0,0,1,0),
                   (0,0,-1,1),
                   (0,-1,-1,0),
                   (0,1,0,0),
                   (-1,0,0,-1),
                   (1,0,0,1)],
    initial_state = (1,0,0,0),
    species = ('b','s','m','A')
    )

delta_t = 1.0
T = np.arange(1.0,80,delta_t)


### OFSP Approximation
from pyme.OFSP import OFSP_Solver

# This stops the system from growing into areas which have zero probability.
def validity_function(X):
	on_off = X[0,:] < 2
	neg_states = np.logical_and.reduce(X >= 0, axis=0)
	return np.multiply(on_off,neg_states)


OFSP_obj = OFSP_Solver(sRNA_model,"SE1",5,1e-6,validity_function)


for t in T:
	OFSP_obj.step(t)
	OFSP_obj.plot(inter=True)			# Interactive mode graphs the marginal each time called.
	OFSP_obj.print_stats

### Hybrid FSP Approximation

"""
The Jacobian of the propensities needed for the higher moments
nabla(propensity func(state)) * state 
"""

def jac_0(x):
	return np.array([0.0,0.0,0.0,0.0])
def jac_2(x):
	return np.multiply(np.array([0.0,0.0,0.0,0.001*x[0]]),x)
def jac_3(x):
	return np.multiply(np.array([0.0,0.0,0.05,0.0]),x)
def jac_4(x):
	return np.multiply(np.array([0.0,0.0,0.02*x[1],0.0]),x)


def Jac_Mat(X,deter_vec,i):
    jac_list = [jac_0,jac_3,jac_4,jac_0,jac_2,jac_0,]
    mat = np.array(map(jac_list[i],X.T)).T
    return mat[deter_vec,:]

stoc_vector = np.array([True,True,False,False])
"""
Gene B and sRNA are considered stochastic.
mRNA and A are considered stochastic.

"""

# evolving the probability forward to gain more regularity.
OFSP_obj = OFSP_Solver(sRNA_model,"SE1",50,1e-6)
OFSP_obj.step(1.0)

####### Hybrid Solver Class ########

from pyme.Hybrid_FSP import Hybrid_FSP_solver

### Hybrid MRCE Computation

MRCE_obj = Hybrid_FSP_solver(sRNA_model,stoc_vector,"MRCE",1e-7,jac=Jac_Mat)
MRCE_obj.set_initial_values(OFSP_obj.domain_states,OFSP_obj.p,t=1.0)

for t in T:
	MRCE_obj.step(t)
	MRCE_obj.print_stats
	MRCE_obj.plot(inter=True)

### Hybrid HL Computation

HL_obj = Hybrid_FSP_solver(sRNA_model,stoc_vector,"HL",1e-7)
HL_obj.set_initial_values(OFSP_obj.domain_states,OFSP_obj.p,t=1.0)
for t in T:
	HL_obj.step(t)
	HL_obj.print_stats