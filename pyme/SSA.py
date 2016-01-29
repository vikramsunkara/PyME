
import numpy as np


def SSA_Windows(model,delta_t,num_steps,initial_start,N):

	Time_Series = []
	Time = []
	for i in range(num_steps):
		if i == 0 :
			Time_Series.append(SSA(model, delta_t, initial_start, N))
			Time.append(delta_t)
		else:
			Time_Series.append(SSA(model, delta_t, initial_start, N,Starting_states=Time_Series[-1]))
			Time.append(Time[-1] + delta_t)
		print(" At time %e"%(Time[-1]))
	return Time_Series, Time



def SSA(model, t_final, initial_start, N, Starting_states = None):

	

	# We only do 10 simulations at a time
	batch = 10

	num_blocks = N/batch

	residual = N - num_blocks*batch

	realisations = np.zeros( (len(initial_start),num_blocks*batch))

	for i in range(num_blocks):
		if Starting_states == None:
			print( SSAsmt(model,batch,0.0,t_final,only_final=True).shape)
			realisations[:,i*batch : (i+1)*batch] = SSAsmt(model,batch,0.0,t_final,only_final=True).T
		else:
			realisations[:,i*batch : (i+1)*batch] = SSAsmt(model,batch,0.0,t_final,only_final=True,starting_states = Starting_states[:,i*batch : (i+1)*batch]).T

	return realisations


## SSAsmt
# This code is for generating simultanuous trajectories of a given CMEPY model.
#@param model CMEPy model format from cmepy.fsp module
#@param N Number of parallel reailsations to be computed
#@param t_start The starting time
#@param t_final The end time all simulations have to reach
def SSAsmt(model,N,t_start,t_final,only_final=False,starting_states = None):

	#import library
	import numpy as np
	import time

	
	# Dimension
	D = len(model.shape)
	R = len(model.transitions)
	
	#inialise Time.
	T = np.zeros((N,1))
	T[:,0] = t_start
	t_min = t_start
	# Now we need to pack the stociometric matrix.
	V = np.zeros((R,D),dtype=np.int)
	for i in range(R):
		V[i,:] = model.transitions[i]
	# N denotes how many parallel session need to run.

	#We give it the initial states
	X = np.zeros((N,D),dtype=np.int)
	if starting_states == None:
		for i in range(N):
			X[i,:] = model.initial_state
	else:
		X[:,:] = starting_states.T
	# Now we are ready!
	
	Store_X = np.zeros((N,D,1),dtype=np.int)
	Store_X[:,:,0] = X[:,:]
	A = np.zeros((N,R))
	A_0 = np.zeros((N,))
	delta_t = np.zeros((N,))
	
	alive_and_not_finished = np.array([True]*N)
	
	while t_min < t_final:
		# calclate the propensities.
		for i in range(R):
			#print("test X "+ str(X))
			A[alive_and_not_finished,i] = map(model.propensities[i], *X[alive_and_not_finished,:].T)
			#A[:,i] = np.where(A[:,i] <0 ,0,A[:,i]) # checking for positivity
		#print("propensities : "+ str(np.nonzero(A[0,:])))
		for i in range(R-1):
			A[:,i+1] +=  A[:,i]
		A_0 = A[:,-1]
		# now A are all weight in 0<= x < 1.

		# now we need to compute our random numbers:
		tau = np.random.random_sample((N,2))
		alive = A_0 > 0
		
		if np.sum(alive) == 0:
			Store_X = np.insert(Store_X,Store_X.shape[2],X,axis=2)
			T = np.insert(T,T.shape[1],t_final,axis=1)
			print(" SSA : Species have died ")
			break
		
		delta_t[alive] = 1/A_0[alive] * np.log(1.0/tau[alive,0])
		delta_t[np.invert(alive)] = 0.0
		if np.sum(delta_t[alive] < 0.0) != 0:
			import pdb
			pdb.set_trace()
		#print( "the shape is "+ str(T.shape[1]))
		T = np.insert(T,T.shape[1],T[:,-1]+delta_t,axis=1)
		
		A_0= A_0*tau[:,1]
		
		
		next_reaction = np.argmax(np.where(A < A_0[:,np.newaxis],0,1),axis=1)
		
		alive_and_not_finished = np.multiply(alive, T[:,-1] <= t_final)
		
		X[alive_and_not_finished,:] += V[next_reaction[alive_and_not_finished],:]

		
		# These are all the reactions.
		Store_X = np.insert(Store_X,Store_X.shape[2],X,axis=2)
		
		t_min = np.min(T[alive,-1])
		
		'''
		ranprint = np.random.rand()
		if ranprint >0.99:
			print(" need to finish %d of %d"%(np.sum(alive_and_not_finished),N))
		'''
		#if __debug__:
		#    print(" SSA is at t: " + str(t_min) + " state " + str(X))
	
	#print("Simulation Has Finished")
	if only_final == False:
		return Store_X, T
	else:
		return Store_X[:,:,-1]
		