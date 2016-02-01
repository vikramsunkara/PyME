"""
The collated code of the tutorial http://github.com/vikramsunkara/PyME/wiki/The-Model-Class
Author: Vikram Sunkara
"""

from pyme.model import Model

# propensities
reaction_1 = lambda *x : 1.0*x[0]
reaction_2 = lambda *x : 1.0*x[1]
reaction_3 = lambda *x : 1.0*x[2]

# transitions
v_1 = (-1,1,0)
v_2 = (0,-1,1)
v_3 = (1,0,-1)

# starting state
x_0 = (10,0,0) # initial populations of the three species.
species_names = ('A','B','C')

# First Model Class
ABC_model = Model( propensities = [reaction_1,reaction_2,reaction_3],
                   transitions = [v_1,v_2,v_3],
                   initial_state = x_0,
                   species = species_names 
                  )

print(ABC_model.propensities)
print(ABC_model.transitions)
print(ABC_model.initial_state)
print(ABC_model.species)

# inside a propensity
print(ABC_model.propensities[0](*ABC_model.initial_state))