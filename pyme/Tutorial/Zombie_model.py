"""
Model file for an example given by Jahnke 2011 
"ON REDUCED MODELS FOR THE CHEMICAL MASTER EQUATION", 
MULTISCALE MODEL. SIMUL, Vol. 9, No. 4,
pp. 1646-1676


Author Vikram Sunkara
"""

import numpy as np 

from pyme.model import Model

# Coefficents

c_1, c_2, c_3, c_4, c_5 = 2.0, 2.5, 0.0, 75.0, 0.2

# Reactions

# S_1 -> *
r1 = lambda *x : c_1*x[0]
v1 = (-1,0)

# S_2 -> *
r2 = lambda *x: c_2 *x[1]
v2 = (0,-1)

# *> S_1
r3 = lambda *x : c_3
v3 = (1,0)

# * -> S_2
r4 = lambda *x : c_4
v4 = (0,1)

#S_1 + S_2 -> S_1 + S_1
r5 = lambda *x : c_5*x[0]*x[1]
v5 = (1,-1)

Zombie_model = Model( propensities = [r1,r2,r4,r5],
						transitions = [v1,v2,v4,v5],
						species = ("S_1 = Zombies","S_2 = People"),
						initial_state = (1,30)
	)

## We run a small OFSP to make the state space more regular.

from pyme.OFSP import OFSP_Solver

Zombie_OFSP = OFSP_Solver(Zombie_model,10,1e-6)

Zombie_OFSP.step(0.1)

Zombie_OFSP.plot()


