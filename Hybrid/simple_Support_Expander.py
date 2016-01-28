# The idea is to simply take an average over a bunch of adjacent states.

# imports
import numpy as np
import pdb
import itertools


def Merge_all(List_SS_combined,list_p,list_Eta,K, N, D_s, D_d):
	
	SS_combined = np.zeros((N, D_s))
	P = np.zeros((N,))
	Eta = np.zeros((D_d,N))
	
	#Fill the matricies for quick sorting.
	start_position = 0
	
	for i in range(K):
		# Stack into matricies.
		if List_SS_combined[i] != "Nil":
			end_position = start_position + List_SS_combined[i].shape[1]
			SS_combined[start_position:end_position,:] = List_SS_combined[i].T
			P[start_position:end_position] = list_p[i][:]
			Eta[:,start_position:end_position] = list_Eta[i][:]
			start_position = end_position
	# Order the states lexicographically	
	indexes =  np.lexsort(SS_combined.T)
	maps = indexes.copy()
	
	# Reindex everything.
	Sorted_SS_C = SS_combined[indexes,:]
	P = P[indexes]
	Eta = Eta[:,indexes]
	
	# looking for similar states.
	diffs = np.where(np.sum(np.abs(Sorted_SS_C[:-1,:]-Sorted_SS_C[1:,:]),axis=1) == 0,1.0,0.0)
	uniques = np.zeros((N,))		
	uniques[1:] = np.where(diffs==0,np.arange(1,N,1),-1)
	repeats = np.sum(diffs)
	temp_diffs = np.zeros((N,))
	temp_diffs[1:] += diffs
	
	for i in range(K-1):
		diffs = np.where(diffs[:-1]*diffs[1:] == 1,1,0)
		#print("diffs in round "+ str(i) + "\n"+str(diffs))
		temp_diffs[(i+2):] += diffs
	for i in range(K-1):
		P[:-(i+1)] +=  (np.where(temp_diffs==(i+1),1.0,0.0)*P)[(i+1):] # Cummulatively adding up.
		Eta[:,:-(i+1)] += (np.where(temp_diffs==(i+1),1.0,0.0)*Eta)[:,(i+1):] # Cummulatively adding up.
		
	unique_arg = np.argsort(uniques)[repeats:]
	positives = P[unique_arg] > 0
	unique_arg = unique_arg[positives]
	
	return Sorted_SS_C[unique_arg,:].T, P[unique_arg], Eta[:,unique_arg]

class Hybrid_N_step_expander(object):
	"""
	Hybrid N-step implementation to expand the domain.
	
	This should apply a simple boundary condition and compute the conditional expectation on the boundary
	"""
	def __init__(self,model,stoc_vector,transitions,propensities,delta_t,eps,core=1):
		self.model = model
		self.delta_t = delta_t
		self.stoc_vector = stoc_vector
		self.eps = eps
		self.core = core
		self.Gtau = 10**(-10) # Not used here
		self.Rtau = 0.0       # Not Used here
		self.Gtau_default = 10**(-10) # Not used here
		
		# testing only
		self.transitions = transitions
		self.propensities = propensities
		
		
		
	def expand(self,**kwargs):
	#Returns epanded domain states
		import lexarrayset
		from cme_matrix import non_neg_states as validity_test
		import itertools
		#from util import Concatenate_state_space as CSS # takes list of matricies.
		#from util import simple_compress as SC # return the compressed vectors and states.
		import time
		
		nabla = kwargs['domain_states']  # state space is D \times N
		return_domain = kwargs['domain_states']
		
		#print("[--Update--] Using the simple Expander " )
		
		#print(" full states \n" + str(nabla[stoc_vector,:]))
		#Start_time = time.time()
		
		N = nabla.shape[1]
		D = nabla.shape[0]
		
		#R = len(self.model.transitions)
		R = len(self.transitions)
			

		deter_vector = np.where(self.stoc_vector == True, False, True)
		
		#if __debug__:
		#	print(" initially :" + str(return_domain.shape))
		
		temp_w = []
		temp_new_states = []
		temp_cond_exp = []

		
		u = kwargs['p']
		
		num_new_states = 0
		
		temp_i =0
		#for transition in self.model.transitions:
		for transition in self.transitions:
			# append current states.
			
			temp_states = nabla + np.array(transition)[:,np.newaxis] 
			
			valid = validity_test(temp_states[self.stoc_vector,:])
			
			new_states_index = lexarrayset.difference_ind(temp_states[self.stoc_vector,:][:,valid], nabla[self.stoc_vector,:])

			# Update the number of terms
			num_new_states += np.sum(new_states_index)
			
			if np.sum(new_states_index) != 0:
				
				##pdb.set_trace()
			  
				# boundary conditbution w
				temp_pre_states = nabla[:,valid][:,new_states_index][deter_vector,:]
				temp_w_terms = np.array([1.0]*len(u[valid][new_states_index]))
				positive_terms = temp_w_terms > 0
				temp_w.append(temp_w_terms[positive_terms])
				temp_new_states.append(temp_states[self.stoc_vector,:][:,valid][:,new_states_index][:,positive_terms]) # new stochastic states
				temp_cond_exp.append(temp_pre_states[:,positive_terms ]) # with transition

			
			else:
				temp_new_states.append('Nil')
				temp_w.append('Nil')
				temp_cond_exp.append('Nil')
				
			temp_i +=1
			
		# Merge all the states and their conditional expectation
		out_new_states, out_w, out_cond_exp = Merge_all( temp_new_states, temp_w, temp_cond_exp, R, num_new_states, np.sum(self.stoc_vector),
np.sum(deter_vector) )
		#pdb.set_trace()
		# Merge the new states with the old.
		
		new_states_restored = np.zeros((D,out_new_states.shape[1]))
		new_u = np.zeros((out_new_states.shape[1],))
		new_u[:] = 0.0 # we can come up with a smarter way to compute this.
		
		new_states_restored[self.stoc_vector,:] = out_new_states[:]
		new_states_restored[deter_vector,:] = np.divide(out_cond_exp[:], out_w[np.newaxis,:]) 
		
		out_states = np.concatenate((return_domain, new_states_restored), axis=1)
		out_u = np.concatenate((u,new_u),axis=0)
			
		#print(" out_states \n" + str(out_states[:].T))
		return out_states, out_states, out_u
