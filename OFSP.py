""" 
OFSP solver. 
Author  : Vikram Sunkara
License GPL V3
"""

import numpy as np
import domain
import state_enum
import FSP.solver


class OFSP_Solver:

	def __init__(self,model,expander_name,compress_window,step_error):
		"""
		@brief OFSP solver for approximation the CME for a given model over time.
		@param model 			: CMEPY Model Class.
		@param expander_name 	: str , "SE1" Simple N-step expander.
		@param compress_window 	: int , number of steps before compressing the domain.
		@param step_error		: float, global error allowed in the sink state per step.
		"""

		self.model				= model
		self.domain_states 		= None
		self.p					= None
		self.t					= 0.0
		self.compress_window 	= compress_window
		self.expander_name		= expander_name
		self.step_error			= step_error


		self._expander			= None
		self._state_enum		= None
		self._solver			= None
		self._steps_to_compress = 0

		self._initialise_state_space_()
		self._set_expander_(0.0)
		self._make_solver()


		#---- Storage ----
		self._stored_t 				= []
		self._stored_p 				= []
		self._stored_domain_states 	= []

		self._probed_t 				= []
		self._probed_probs 			= []
		self._probed_states			= []

	def _initialise_state_space_(self):
		self.domain_states = domain.from_iter((self.model.initial_state,))
		self._state_enum = state_enum.StateEnum(self.domain_states)
		self.p = self._state_enum.pack_distribution({self.model.initial_state:1.0})

	def _set_expander_(self,h):
		if self.expander_name == "SE1": # N-step expander
			from FSP.simple_expander import SimpleExpander
			self._expander = SimpleExpander(self.model.transitions, depth=1,)
		elif Expander =='GORDE' : # Gated one reaction expander.
			from FSP.GORDExpander import GORDE_Algo as GORDE
			self._expander = GORDE(self.model,h,self.max_error_per_step)

	def _make_solver(self):
		self._solver=FSP.solver.create(
		self.model,
		self.domain_states,
		self._state_enum,
		self._expander,
		p_0 = self.p,
		t_0 = self.t
		)

	def _compress_domain(self):
		from FSP.sunkarautil import simple_compress

		self.domain_states,self.p = simple_compress(self._solver.domain_states.T,self._solver.y[0],self.step_error*0.5)
		self.domain_states = self.domain_states.T

		self._state_enum = state_enum.StateEnum(self.domain_states)
		order = np.lexsort(self.domain_states)
		self.p = self.p[order]
		self._steps_to_compress = 0
		self._make_solver

	def step(self,t):
		"""
		@brief step : evolves the density forward to the time point t. ..warning : The underlying scipy ODE solver is not great, please take small time steps.
		@param t : float  

		"""
		self._set_expander_(t-self.t)
		
		self._solver.step(t,self.step_error)
		self._steps_to_compress += 1
		self.t = t

		if self._steps_to_compress == self.compress_window :
			self._compress_domain()

		self.domain_states  = self._solver.domain_states
		self.p 				= self._solver.y[0]
		self._state_enum	= self._solver.domain_enum # domain hashing class.


	def plot(self,inter=False):
		"""
		@param inter :(Boolean) Interactive mode. If true, then the picture is redrawn in the exisiting figure.
		"""
		from plotters import plot_marginals
		plot_marginals(self.domain_states.T,self.p,"OFSP Using :"+self.expander_name,self.t,labels=self.model.species,interactive=inter)

	def set_initial_states(self,domain_states,p):
		"""
		@brief initialise the solver if the initial density is not a point mass.
		@param domain_states 	: numpy.ndarray, shape = (num of species x num of states)
		@param p 				: numpy.ndarray, shape = (num of states,)
		"""
		self.t = t
		if domain_states.shape[1] == p.shape[0] :
			domain_states = domain_states.T
		order = np.lexsort(domain_states)       
		self.p = p[order].flatten()
		self.domain_states = domain_states[:,order]
		self.initial_state_enum = state_enum.StateEnum(self.domain_states)

		self._make_solver()
	@property
	def print_stats(self):
		print(" t : %6.4f | states : %4d | prob(in sink) : %4.3e | Steps to Compress : %3d " 
				% (self.t,self.domain_states.shape[1],1-np.sum(self.p),self.compress_window-self._steps_to_compress)) 
	def check_point(self):

		self._stored_domain_states.append(self.domain_states)
		self._stored_t.append(self.t)
		self._stored_p.append(self.p)


	### WARNING TO NOT USE, PYTHON STILL DOES NOT KNOW HOW TO STORE THE PROPENSITY FUNCTIONS IN THE MODEL object.
	def stash(self,location,name):
		"""
		..warning not working yet.
		"""
		import pickle
		f = open(location+name+".pck",'wb')
		pickle.dump(self,f)
		f.close()

		print("[--Update--] The solver object and its content have been saved to %s"%(location + name))

	def plot_checked(self):
		import pylab as pl
		pl.ioff()
		from statistics import expectation
		exp = []

		# The expectation plotter
		if len(self._stored_t) != 0:
			pl.figure(2)
			pl.title(" Method %s"%("OFSP"))
			pl.xlabel("Time, t")
			pl.ylabel("Expectation")

			for i in range(len(self._stored_t)):
				exp.append(expectation((self._stored_domain_states[i],self._stored_p[i])))

			EXP = np.array(exp).T

			for i in range(EXP.shape[0]):
				pl.plot(self._stored_t,EXP[i,:],'x-',label=self.model.species[i])
			pl.legend()

		# The probability plotter
		if len(self._probed_t) != 0:
			pl.figure(3)
			pl.title(" Method %s | Probing States over Time "%("OFSP"))
			pl.xlabel("Time, t")
			pl.ylabel("Probability")

			probs = np.array(self._probed_probs).T

			for i in range(probs.shape[0]):
				pl.plot(self._probed_t,probs[i,:],'x-',label=str(self._probed_states[0][:,i]))
			pl.legend()

		pl.show()

	def probe_states(self,X):
		"""
		@brief probe_states when called a set of states the corresponding probabilties are stored away. Later can be viewed using (self.plot_checked())
		@param X :numpy.ndarray,  shape = (num species, num states) 
		"""

		# We use the hash class to find the positions of the states to probe in the domain_states.
		
		non_zero_probs = self._state_enum.contains(X)
		
		probed_probs = [0.0]*X.shape[1]

		if np.sum(non_zero_probs) != 0:
			 # pickup only the states which are in the state space
			positions = self._state_enum.indices(X[:,non_zero_probs])
			_counter = 0
			for i in range(X.shape[1]):
				if non_zero_probs[i] == True:

					if np.sum(self.domain_states[:,positions[_counter]] - X[:,i]) == 0:
						probed_probs[i] = self.p[positions[_counter]]
						_counter += 1

		self._probed_probs.append(probed_probs)
		self._probed_states.append(X)
		self._probed_t.append(self.t)



