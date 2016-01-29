'''
Vikram Sunkara. Hybrid FSP modules.

'''
# imports
import numpy as np
import pdb
import itertools
import state_enum
import scipy.sparse
import time

import pylab as pl


def Hybrid_FSP(model, X, w, h, position, valid, stoc_vector, expander, tau, residue):
	# We take a false step and if there is soo much probability lost, we grow the domain.
	
	from scipy.integrate import ode
	from Support_Expander import positions
	
	from n_term import simple_compress as compress
	#from n_term import compress_by_marginal as compress
	
	
	from Compute_A import compute_A
	
	# Growth Flag
	grew = False
	
	Energy = np.sum(w)
	
	new_p_0 = np.zeros((X.shape[1],))
	
	new_p_0 = w[:]
		
	def f_sparse(t,y,A):
			return A.dot(y)
	A_sparse = compute_A(X, model.propensities, position, valid)
	ode_obj = ode(f_sparse).set_integrator('lsoda',method='bdf')
	ode_obj.set_initial_value(new_p_0, 0).set_f_params(A_sparse)
	
	#pdb.set_trace()
	while ode_obj.successful() and ode_obj.t < h:
		ode_obj.integrate(h)
	sink = Energy - np.sum(ode_obj.y)
	
	if sink + residue > tau:
		# there is soo much outflow, we need to grow.
		
		residue = 0.0 # reset the residual
		#pdb.set_trace()
		new_X, garbage1, garbage2  = expander.expand(domain_states = X, p = w)
		
		# new domain enum.
		
		new_domain_enum = state_enum.StateEnum(new_X,stoc_vector)
		# now we need new position and valid vectors and a new probability vector.
		#print(" Growing the Hybrid Domain by %d states."%(new_X.shape[1] - X.shape[1]) )
		
		grew = True
		
		new_X = new_domain_enum.ordered_full_dom
		
		new_w = np.zeros((new_X.shape[1],))  # need to add one state for the sink state.
		old_domain_indices = new_domain_enum.indices(X)
		
		new_w[old_domain_indices] = w
		
		valid, position  = positions(new_X, model.transitions, stoc_vector,new_domain_enum)
		
		A_sparse = compute_A(new_X, model.propensities, position, valid)
		
		ode_obj = ode(f_sparse).set_integrator('lsoda',method='bdf')
		ode_obj.set_initial_value(new_w, 0).set_f_params(A_sparse)

		while ode_obj.successful() and ode_obj.t < h:
			ode_obj.integrate(h)
		
		sink = Energy - np.sum(ode_obj.y)
		
		'''
		if sink > tau:
			#raise ValueError(str(ode_obj.y[-1]) + " flowed out, Tau :" + str(tau)) 
			print(" Too much out flow : "+  str(sink) + " flowed out, Tau :" + str(tau) )
		'''	
		# We need to remove some redundant states.
		
		#pdb.set_trace()
		w_dot = f_sparse(1, ode_obj.y, A_sparse)
		
		# where the derivative is zero.
		#negative_dot = (w_dot <= 0)[:-1]
		# REWRITE THE NEGATIVE DOT COMPRESSION DOESNT WORK
		negative_dot = np.array([True]*(len(w_dot)-1))
		positive_dot = np.invert(negative_dot)
		

		
		## Baised compress
		#keep_states, keep_prob = compress( new_X[:,negative_dot],ode_obj.y[:-1][negative_dot], 0.0*tau) # No reduction.
		keep_states, keep_prob = compress( new_X[:,negative_dot],ode_obj.y[:-1][negative_dot], tau)
		
		## non biased compress##
		#shrunck_X, shrunck_w = compress( new_X,ode_obj.y, 0.0001*tau)
		
		#pdb.set_trace()
		if keep_states.shape[1] < np.sum(negative_dot):
		#if shrunck_X.shape[1] < new_X.shape[1]:

			#print(" Reducing the Hybrid Domain by : %d states."%(np.sum(negative_dot) - keep_states.shape[1]))
			
			# we have a reduction, so we adjust the indicies.
			shrunck_X = np.concatenate((new_X[:,positive_dot],keep_states), axis=1)
			
			shrunck_w = np.concatenate((ode_obj.y[positive_dot],keep_prob))
			
			# need to reindex everything and generate new positions.
			
			new_domain_enum = state_enum.StateEnum(shrunck_X,stoc_vector)
			
			new_X = new_domain_enum.ordered_full_dom

			new_w = shrunck_w[new_domain_enum.order]
			
			valid, position  = positions(new_X, model.transitions, stoc_vector,new_domain_enum)
			
			wdot = w_dot


		else:

			#print('No Reduction')
			new_w = ode_obj.y
	else:
		#residue += ode_obj.y[-1]
		residue += sink
		new_X = X.copy()
		new_w = ode_obj.y

		new_domain_enum = False
	
	
	#print("existing with residue" + str(residue))
	return new_X , new_w, position, valid, residue, grew, new_domain_enum
