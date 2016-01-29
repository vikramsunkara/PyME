'''
Hybrid FSP solver
Author: Vikram Sunkara
License GPL V3
'''


import numpy as np
import Hybrid.state_enum

class Hybrid_FSP_solver:

	def __init__(self,model,stoc_vector,model_name,sink_error,jac=None):
		"""
		@brief Hybrid_FSP_Solver which can compute MRCE and HL hybrid approximations for a given model. 

		@param model 		: model.Model. 
		@param stoc_vector 	: numpy.ndarray.  
		@param model_name 	: str, 'MRCE' or 'HL'.
		@param sink_error 	: float, maximum error allowed in the sink state.
		@param jac 			: (List of Functions), The jacobian of the propensity functions.
		
		@return Hybrid_FSP_Solver	
		"""

		#### External Variables ####
		self.model 			= model
		self.model_name 	= model_name
		self.stoc_vector 	= stoc_vector
		self.deter_vector	= np.invert(stoc_vector)
		self.model 			= model
		self.sink_error		= sink_error
		self.t 				= 0.0
		self.Jac 			= jac


		#### Internal Variavles ####
		self._N_s 			= len(model.transitions[0])
		self._N_r			= len(model.transitions)
		self._positions 	= None
		self._valid			= None
		self.domain_enum	= None
		self._expander		= None
		self._new_model		= None
		self._X				= None
		self._w 			= None

		self._V_s			= None
		self._V_d			= None
		self._new_prop_s	= None
		self._stoc_positions	= None

		self.__residual		= 0.0
		self.__expanded 	= 0
		self.__HNE			= None
		self.__Implicit_solver = None
		self._stored_t		= []
		self._stored_X		= []
		self._stored_w		= []

		self._probed_probs	= []
		self._probed_states	= []
		self._probed_t 		= []

		# Initialising functions.

		self._set_ODE_integrator_
		self._make_splitting

	@property
	def _make_splitting(self):

		from Hybrid.Support_Expander import stochiometric_split
		from Hybrid.Support_Expander import positions

		from model import Model

		if self._X == None:

			self._X = np.zeros((self._N_s,1))
			self._X[:,0] = self.model.initial_state
		
			self._w = np.zeros((1,))
			self._w[0] = 1.0

		self.V = []
		for i in range(self._N_r):
			self.V.append(np.array(self.model.transitions[i]))
		
		# make new_stochiometry 
		self._V_d, self._V_s, self._stoc_positions, self._new_prop_s = stochiometric_split(self.V, self.model.propensities, self.deter_vector)

		self._new_model = Model(
								propensities = self._new_prop_s,
								transitions= self._V_s,
								shape= self.model.shape,
								initial_state= self.model.initial_state)

		# initialising the expander.
		self._expander =  self.__HNE(self._new_model,self.stoc_vector,self.model.transitions,self.model.propensities,1.0,1e-6)

		self.domain_enum = Hybrid.state_enum.StateEnum(self._X,self.stoc_vector)
		self._valid, self._position = positions(self._X, self._V_s, self.stoc_vector, self.domain_enum)

	@property
	def _set_ODE_integrator_(self):
		if self.model_name == 'HL' :

			from Hybrid.implicit_ODE_HL_ODE import implicit_black_box as Implicit_solver
			from Hybrid.Support_Expander import Hybrid_HL_N_step_expander as HNE
			self.Jac = None

		else:
			if self.Jac == None:
				from Hybrid.implicit_ODE import implicit_black_box as Implicit_solver
			else:
				from Hybrid.implicit_ODE_With_Jac import implicit_black_box as Implicit_solver

			from Hybrid.simple_Support_Expander import Hybrid_N_step_expander as HNE
		
		self.__HNE = HNE
		self.__Implicit_solver = Implicit_solver


	def set_initial_values(self,domain_states,p,t = 0.0):
		"""
		@brief initialise the hybrid solver with the OFSP state space and probabiltiy. The function will automatically do the projection.
		@param domain_states 	: numpy.ndarray, shape = (num of species x num of states)
		@param p 				: numpy.ndarray, shape = (num of states,)
		@param t 				: numpy.ndarray, Default = 0.0
		"""

		from Hybrid.Support_Expander import positions

		# We project them down

		from Hybrid.proj import project_to_Hybrid as POH
		from statistics import expectation

		clean = p > 1e-9
		domain_states = domain_states[:,clean]
		p = p[clean]

		self.t = t

		self._X, self._w = POH(domain_states,p,self.stoc_vector)

		if self.model_name == "HL":
			self._X[self.deter_vector,:] =  expectation((domain_states,p))[self.deter_vector,np.newaxis]

		# These are the key hashing components needed to find adjacent states.

		self.domain_enum = Hybrid.state_enum.StateEnum(self._X,self.stoc_vector)
		self._valid, self._position = positions(self._X, self._V_s, self.stoc_vector, self.domain_enum)

	def step(self,t_final,sink_error=None,fine_res=1):
		"""
		@brief step function which evolves the solver forward.
		@param t_final 		: float, The time point to evolve the solver to
		@param sink_error 	: float, By defult is the chosen at the start, however, it can be adjusted in between steps.
		@param fine_res 	: int, [Caution] Feature not working yet. 
		"""

		from Hybrid.Hybrid_FSP import Hybrid_FSP

		delta_t = t_final-self.t


		if t_final == self.t:
			return 0

		## find delta_h 
		steps = np.zeros((self._X.shape[1],))
		for i in range(len(self._new_prop_s)):
			steps += map(self._new_prop_s[i],* self._X )

		h = np.minimum(fine_res/np.amax(steps),delta_t)
		# internal time within the step delta_t
		t = 0.0

		# Strang Splitting Scheme
		while t < delta_t:


			# Evolving the conditional moments
			for iterer in range(int(fine_res)):
				self._X = self.__Implicit_solver(self.model.propensities,self.V,
									self._X, self._w, h/(2.0*fine_res),
									self.deter_vector,self._stoc_positions, 
									self._position, self._valid, self.Jac)

			self._X, self._w, self._position, self._valid, self.__residual, grew, temp_domain_enum = Hybrid_FSP(self._new_model, 
																									self._X, self._w, h, 
																									self._position, self._valid, self.stoc_vector,
																									 self._expander, self.sink_error, self.__residual)
			for iterer in range(int(fine_res)):
				self._X = self.__Implicit_solver(self.model.propensities,self.V,
									self._X, self._w, h/(2.0*fine_res),
									self.deter_vector,self._stoc_positions, 
									self._position, self._valid, self.Jac)
			t += h
			self.t += h
			if grew == True:
				self.__expanded += 1
				self.domain_enum = temp_domain_enum

			# recompute h
			steps = np.zeros((self._X.shape[1],))
			for i in range(len(self._new_prop_s)):
				steps += map(self._new_prop_s[i],* self._X )

			h = np.minimum(fine_res/np.amax(steps),delta_t-t)

	@property
	def print_stats(self):
		print(" t : %6.4f | states : %4d | prob(l) : %4.3e | residue in Sink : %4.2e  | Expanded : %3d | Mean: %s" 
				% (self.t,self._X.shape[1],1-np.sum(self._w),self.__residual, self.__expanded,np.sum(np.multiply(self._X,self._w),axis=1))) 
	@property
	def domain_states(self):
		return self._X

	@property
	def p(self):
		return self._w

	def plot(self,inter= False):
		"""
		@param inter :bool, Interactive mode. If true, then the picture is redrawn in the exisiting figure.
		"""
		from plotters import plot_marginals
		plot_marginals(self._X.T,self._w,"Hybrid Using :"+self.model_name,self.t,labels=self.model.species,interactive=inter)

	
	def check_point(self):
		"""
		@brief check_point stores the current state space and the marginal probabilty.
		"""

		self._stored_X.append(self._X)
		self._stored_t.append(self.t)
		self._stored_w.append(self._w)

	def plot_checked(self):
		"""
		@brief plot_checked plots the expectations of the data check pointed.
		"""
		import pylab as pl
		pl.ioff()
		
		if len(self._stored_t) != 0:
			pl.figure(2)
			pl.title(" Method %s"%(self.model_name))
			pl.xlabel("Time,t")
			pl.ylabel("Expectation")

			exp = []

			for i in range(len(self._stored_t)):
				exp.append(np.sum(np.multiply(self._stored_X[i],self._stored_w[i]),axis=1))
			
			EXP = np.array(exp).T


			for i in range(EXP.shape[0]):
				pl.plot(self._stored_t,EXP[i,:],'x-',label=self.model.species[i])

			pl.legend()

		# The probability plotter
		if len(self._probed_t) != 0:
			pl.figure(3)
			pl.title(" Method %s | Probing States over Time "%(self.model_name))
			pl.xlabel("Time, t")
			pl.ylabel("Marginal Probability")

			probs = np.array(self._probed_probs).T

			for i in range(probs.shape[0]):
				pl.plot(self._probed_t,probs[i,:],'x-',label=str(self._probed_states[0][self.stoc_vector,i]))
			pl.legend()
		pl.show()

	def probe_states(self,X):
		"""
		@brief probe_states when called a set of states the corresponding probabilties are stored away. Later can be viewed using (self.plot_checked())
		@param X :numpy.ndarray,  shape = (num species, num states) 
		"""

		non_zero_probs = self.domain_enum.contains(X)
		
		probed_probs = [0.0]*X.shape[1]

		if np.sum(non_zero_probs) != 0:
			 # pickup only the states which are in the state space
			positions = self.domain_enum.indices(X[:,non_zero_probs])
			_counter = 0
			for i in range(X.shape[1]):
				if non_zero_probs[i] == True:
					if np.sum(self._X[self.stoc_vector,positions[_counter]] - X[self.stoc_vector,i]) == 0:
						probed_probs[i] = self._w[positions[_counter]]
						_counter += 1



		self._probed_probs.append(probed_probs)
		self._probed_states.append(X)
		self._probed_t.append(self.t)


