"""
Collection of utility functions to do tasks for CMEPY
Author :Vikram SUnkara
"""
import numpy as np
import scipy as sp



##The following function will concatenate the states and probabilities from lists of statespaces and probabilties.
#@param L_state_space We need to get a list of all the state space
#@param L_prob list of all the probabilties.	
#@return - state_space_new : the new state space with all uniques states
#	    - p_new : the new probability vector with positions corresponding to the state_space_new.
def Concatenate_state_space(L_state_space,L_prob):
	Nils = 0
	K = len(L_state_space)
	
	# concatenate them into a single big problem
	SS_combined = np.concatenate(L_state_space,axis=0)
	P = np.concatenate(L_prob,axis=0)
	N = SS_combined.shape[0]
	
	indexes =  np.lexsort(SS_combined.T)
	
	maps = indexes.copy()
	
	Sorted_SS_C = SS_combined[indexes,:]
	P = P[indexes]
	
	#print("weird \n "+ str(Sorted_SS_C[:-1,:]))
	#print("weirder \n "+ str(Sorted_SS_C[1:,:]))
	
	diffs = np.where(np.sum(np.abs(Sorted_SS_C[:-1,:]-Sorted_SS_C[1:,:]),axis=1) == 0,1.0,0.0)
	
	uniques = np.zeros((N,))		
	uniques[1:] = np.where(diffs==0,np.arange(1,N,1),-1)
	
	repeats = np.sum(diffs)
	
	temp_diffs = np.zeros((N,))
	temp_diffs[1:] += diffs
	#print("first pass \n"+ str(temp_diffs))
	# now we need to add it up
	
	for i in range(K-Nils-1):
		diffs = np.where(diffs[:-1]*diffs[1:] == 1,1,0)
		#print("diffs in round "+ str(i) + "\n"+str(diffs))
		temp_diffs[(i+2):] += diffs
		
	#print(str(Sorted_SS_C))	
	#print(str(temp_diffs)) 
	
	#print("ORIGINAL P-----\n"+ str(P))
	# need to fill the p properly.
	for i in range(K-Nils-1):
		#print(np.where(temp_diffs==(i+1),1.0,0.0)*P)
		P[:-(i+1)] +=  (np.where(temp_diffs==(i+1),1.0,0.0)*P)[(i+1):]
		#print("P----\n"+ str(P))
		
	# NOw we need to take the uniques
	
	unique_arg = np.argsort(uniques)[repeats:]
	
	
	return Sorted_SS_C[unique_arg,:], P[unique_arg]

	

##COMPUTE THE MEAN OF A STATE SPACES WITH THE PROBABILTIES.
#@param states : the state space shape: (Number of Species X Number of states).
#@param p : as a probability vector.
#@return : mu expectation as a vector of all the species.
def expectation(states,p):
	weighted_states = states * p[np.newaxis, :]
	mu = np.add.reduce(weighted_states, axis=1)
	return mu
	
##Function to give an average over the state spaces.
#@param states the state space in row vectors:  (Number of Species X Number of states)
#@return mu average of the state space.
def averaging(states):
	return np.sum(states,axis=0)/np.array([states.shape[0]]*states.shape[1]).astype(float)

##Compressing algorithm of th OFSP using the Best N-terms approximation in the ell_1 norm.
#@param fsp_solver Fsp solver object
#@param DtateEnum The domain indexing class
#@param epsilon  The amount to compress by in the ell**1 norm
#@param t time at which this is happening.
#return	fsp_solver : New solver with the intial condition given by the new state space and probability vector.
def compress_solver(fsp_solver,StateEnum,epsilon,t):
	if not (0.0 <= epsilon <= 1.0):
        	raise ValueError('epsilon must be within range: 0.0 <= epsilon <= 1.0')

	if len(fsp_solver.domain_states[1]) > 1:
		# create array representation of distribution
		states = fsp_solver.domain_states.T
		probabilities, p_sink = fsp_solver.y
		
		probabilities = np.abs(probabilities)
		
		# order entries with respect to increasing probability
		order = np.argsort(probabilities)
		states = states[order]
		probabilities = probabilities[order]

		# discard the largest number of states while keeping the
		# corresponding net probability discarded below epsilon
		cumulative_probability = np.add.accumulate(probabilities)
		approximation = (cumulative_probability >= epsilon)
		states = states[approximation]
		probabilities = probabilities[approximation]
		if states.shape[0] == 0:
			return 0
		fsp_solver.domain_states = states.T
		# Now we need to start a new solver.
		new_state_enum = StateEnum(states.T)
		fsp_solver.domain_enum = new_state_enum
	    	
	    	#print("WOW ABOUT TO BLOW UP on CORE "+ str(rank) + " we have "+ str(probabilities.shape) + " " + str(fsp_solver.domain_states.shape) + " " + str(states.shape))
	    	
	    	fsp_solver.solver.restore(
              		p_0 = probabilities,
              		sink_0 = p_sink + epsilon,
              		domain_states = fsp_solver.domain_states,
              		domain_enum = fsp_solver.domain_enum,
              			)
              	fsp_solver.solver.set_restore_point()
		
		#fsp_solver=create(
			#model,
			#states.T,
			#new_state_enum,
			#expander,
			#p_0 = probabilities,
			#t_0 = t
			#)
	#return fsp_solver

		
##Computing the marginal distribution given the state space matrix and the probability vector
#@param state_space: The State space which is a (WARNING:) N * D matrix. D is the dimension of the problem
#@param p  N * 1 vector of postive values below 1.
#@param Name of file, it has to be a string.
#@param t time point of the data
#@param labels the labels you want to add to the subgraphs
def plot_marginals(state_space,p,name,t,labels = False):
	import matplotlib
	#matplotlib.use("PDF")
	#matplotlib.rcParams['figure.figsize'] = 5,10
	import matplotlib.pyplot as pl
	pl.suptitle("time: "+ str(t)+" units")
	print("time : "+ str(t))
	
	D = state_space.shape[1]

	for i in range(D):
		marg_X = np.unique(state_space[:,i])
		A = np.where(marg_X[:,np.newaxis] == state_space[:,i].T[np.newaxis,:],1,0)
		marg_p = np.dot(A,p)
		pl.subplot(int(D/2)+1,2,i+1)
		pl.plot(marg_X,marg_p)
		pl.axvline(np.sum(marg_X*marg_p),color= 'r')
		pl.axvline(marg_X[np.argmax(marg_p)],color='g')
		if labels == False:
			pl.xlabel("Specie: " + str(i+1))
		else:
			pl.xlabel(labels[i])
	#pl.savefig("Visuals/marginal_"+name+".pdf",format='pdf')
	pl.show()
	pl.clf()

##Simple Compress : best N-term approximation under the ell_1 norm
#@param state_space the state space shape: (Number of Species X Number of states) 
#@param p probability vector
#@param eps the ell_1 error to remove
#@return -Compressed state space
#	    -Compressed Probs
def simple_compress(state_space,p,eps):
	
	# we will sort it and add up accumulatively and truncate
	arg_sort_p = np.argsort(p)
	a_s_p_a = np.add.accumulate(p[arg_sort_p])
	remove_num = np.sum(np.where(a_s_p_a < eps,1,0))
	
	return state_space[arg_sort_p[remove_num:],:], p[arg_sort_p[remove_num:]]
	
## Gets a list of all the get_marginals
#@param states it is a N_s \times N states space
#@param p is the probability vector
#@return - List of each marginal state space
#			- List of each marginal probaiblity.

def get_marginals(states,p):
    state_space = states.T
    D = state_space.shape[1]
    marg_X_list = []
    marg_p_list = []
    for i in range(D):
        marg_X_list.append(np.unique(state_space[:,i]))
        A = np.where(marg_X_list[-1][:,np.newaxis] == state_space[:,i].T[np.newaxis,:],1,0)
        marg_p_list.append(np.dot(A,p))
    return marg_X_list, marg_p_list


def plot_2D_heat_map(states,p,labels):
	import pylab as pl
	X = np.unique(states[0,:])
	Y = np.unique(states[1,:])
	X_len = len(X)
	Y_len = len(Y)
	Z = np.zeros((X.max()+1,Y.max()+1))
	for i in range(len(p)):
		Z[states[0,i],states[1,i]] = p[i]
	pl.clf()	
	pl.imshow(Z.T, origin='lower')
	pl.xlabel(labels[0])
	pl.ylabel(labels[1])
	pl.draw()
	#pl.show()
