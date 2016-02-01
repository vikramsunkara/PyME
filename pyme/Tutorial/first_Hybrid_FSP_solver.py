"""
First hybrid FSP solver tutorial using the Jahnke Model.
Author Vikram Sunkara
"""
import numpy as np

#import model
from Zombie_model import * 


# import Hybrid Solver

from pyme.Hybrid_FSP import Hybrid_FSP_solver

"""
("S_1 = Zombies","S_2 = People")
* S_1 is considered stochastic 
* S_2 will be approximated by marginal distributions
"""
stoc_vector = np.array([True,False]) 

# initialising the Hybrid Solver
Zombie_solver = Hybrid_FSP_solver(Zombie_model,stoc_vector,"MRCE",1e-7)
Zombie_solver.set_initial_values(Zombie_OFSP.domain_states,Zombie_OFSP.p,t=Zombie_OFSP.t)


T = np.arange(Zombie_OFSP.t,2.0,0.1)

for t in T:
	Zombie_solver.step(t)
	Zombie_solver.print_stats

Zombie_solver.plot()







