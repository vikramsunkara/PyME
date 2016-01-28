import numpy as np
import scipy as sp

''' We give the function the state space and mean and it returns the list of proportions of probability we should hand out to each cluster 

Notes: ALL inputs and outputs have to be numpy matricies.

INPUTS:
	state_space = N \times D matrix, N number of states and D is the dimension of a state
	mu = K \times D, M is the number of means/cores, D is the dimension of a state
	tau is the number less then 1. Is the threshold after which the proportion is turned to zero. It is a ratio of the distances, so if the ratio of the distance to one mean is is less that tau then we change its proportion to zero.
	
Outputs:
	Proportions = N \times K matrix. The proportion of probability to be allocated to each of the clusters.
'''
def partition_algo_distances(state_space,mu,tau):
	
	K = mu.shape[0]
	
	distance = np.sum(np.power(state_space[:,:,np.newaxis] - mu.T[np.newaxis,:,:],2 ),axis=1)
	
	distance = distance +1 # this way we can devide by the distances to get proper proportions.
	
	inv_dist = 1.0/np.power(distance,1)
	
	
	max_inv_dist = np.amax(inv_dist,axis=1)
	# need to draw each one out and delete it.
	for k in range(K):
		edges = np.where( inv_dist[:,k] == max_inv_dist,max_inv_dist,0.0) 
		inv_dist[:,k] = np.where( (inv_dist[:,k] < tau), 0.0,inv_dist[:,k]) + np.where(edges > tau ,0.0,edges)
	sum_to_divide = np.sum(inv_dist,axis = 1)
	
	proportions = inv_dist/sum_to_divide[:,np.newaxis]
	
	return proportions

''' 
This function will give a very tight clustering with respect to simply the l**1 norm. 

Notes: ALL inputs and outputs have to be numpy matricies.

INPUTS:
	state_space = N \times D matrix, N number of states and D is the dimension of a state
	mu = K \times D, M is the number of means/cores, D is the dimension of a state
	tau is the number less then 1. Is the threshold after which the proportion is turned to zero. It is a ratio of the distances, so if the ratio of the distance to one mean is is less that tau then we change its proportion to zero.
	
Outputs:
	Proportions = N \times K matrix. The proportion of probability to be allocated to each of the clusters.
'''
def partition_algo_distances_tight(state_space,mu,tau):
	
	K = mu.shape[0]
	
	distance = np.sqrt(np.sum(np.power(state_space[:,:,np.newaxis] - mu.T[np.newaxis,:,:],2 ),axis=1))

	arg_min_dist = np.argmin(distance,axis=1)

	for k in range(K):
		distance[:,k] = np.where( arg_min_dist == k ,1.0,0.0)
	
	
	return distance




''' 
This function will give a very tight and stingy clustering with respect to simply the l**1 norm. 

Notes: ALL inputs and outputs have to be numpy matricies.

INPUTS:
	state_space = N \times D matrix, N number of states and D is the dimension of a state
	mu = K \times D, M is the number of means/cores, D is the dimension of a state
	tau is the number less then 1. Is the threshold after which the proportion is turned to zero. It is a ratio of the distances, so if the ratio of the distance to one mean is is less that tau then we change its proportion to zero.
	
Outputs:
	Proportions = N \times K matrix. The proportion of probability to be allocated to each of the clusters.
'''
def partition_algo_distances_tight_stingy(state_space,mu,tau,rank):
	
	K = mu.shape[0]
	
	mu = np.floor(mu)
	#print(" In STINGY Means " + str(mu))
	
	distance = np.abs(np.sum(state_space[:,:,np.newaxis] - mu.T[np.newaxis,:,:],axis=1))
	
	#distance = distance # this way we can devide by the distances to get proper proportions
	#print("Crazy test core "+str(rank)+ " " + str(np.sum(distance)))
	max_inv_dist = np.amin(distance,axis=1)
	#print("Crazy------------------------------>"+str(rank)+" ------- " + str(np.sum(max_inv_dist)))
	# need to draw each one out and delete it.
	# this algorithm picks the closest one to go to the core. So it is send to only a single core.
	#temp_test = []
	#temp_test_2 = []
	for k in range(K):
		#temp_test.append(np.sum(distance[:,k]))
		distance[:,k] = np.where( distance[:,k] == max_inv_dist,1.0,0.0)
		#temp_test_2.append(np.sum(distance[:,k]))
	#sum_to_divide = np.sum(inv_dist,axis = 1)
	#print(" IN STINGy " + str(rank) + " ------------------------ " + str(temp_test))
	#print(" IN STINGYY" + str(rank) + " ------------------------" + str(temp_test_2))
	# now we recheck to see if we keep any.
	
	### remove this
	#temp_test = []
	#for i in range(K):
	#	temp_test.append(np.sum(distance[:,k]))
	#print(" IN STINGY "  + str(rank) + " we have " + str(temp_test))
	#temp = np.where(inv_dist[:,rank-1] == 1,1,0)
	if np.sum(distance[:,rank-1]) < 3000:
		distance[:,:] = 0
		distance[:,rank-1] = 1
	
	#proportions = inv_dist/sum_to_divide[:,np.newaxis]
	
	return distance



'''
The follwoing function will help segrigate the state space to its rightfull cores

Inputs:  state_space : state_space = N \times D matrix, N number of states and D is the dimension of a state
	 Proportions = N \times K matrix. The proportion of probability to be allocated to each of the clusters.
	 p : N \times 1 vector which will have the probabilties of the corresponding states in state space.
Output:
	sub_state_spaces : ( M \times D ) \times K : List of the state space going out to each core
	sub_probs :  ( M \times 1 ) \times K : List of the probabilities going out to each core
'''
def seperate_via_proportions(state_space,proportions, p):

	
	
	N = state_space.shape[0]
	K = proportions.shape[1]
	
	'''
	Testing this out 
	
	temp = np.sum(np.sum(proportions,axis=1))
	if temp != N:
		print( "\n \n \n  FOUND ITTTTTTT NOT CORRECT PROPORTIONS "+ str(temp) + "should be "+ str(N)+" \n\n\n")	
	else:
		print("\n \n \n They Match UP \n\n\n")
	'''
	
	sub_state_spaces = []
	sub_probs = []
	
	for k in range(K):
		if np.sum(proportions[:,k])!= 0:
			# we can fills it
			temp_position = np.where(proportions[:,k] == 1,range(N),-1)
			args = np.argsort(temp_position)
			args = args[N-np.sum(proportions[:,k]):]
			sub_state_spaces.append(state_space[args,:])
			sub_probs.append(p[args])
		else:
			sub_state_spaces.append("Nil")
			sub_probs.append("Nil")
	return sub_state_spaces,sub_probs
		
'''
The following function will help seperate the state space using the K-mean method from Scipy.

Inputs: state_space : state_space = N \times D matrix, N number of states and D is the dimension of a state.
        p : N \times 1 vector which will have the probabilties of the corresponding states in state space.
	K : number of cluster we want to split it up into.
	tau is the number less then 1. Is the threshold after which the proportion is turned to zero. It is a ratio of the distances, so if the ratio of the distance to one mean is is less that tau then we change its proportion to zero.
	
Output:
	sub_state_spaces : ( M \times D ) \times K : List of the state space going out to each core.
	sub_probs :  ( M \times 1 ) \times K : List of the probabilities going out to each core.
'''
def seperate_via_kmeans(state_space,p,K,tau=0.1):
	from scipy.cluster.vq import kmeans	
	#centres= np.floor(kmeans2(state_space,K)[0]) # these are the original lines
	
	# the following are being added as hacks
	
	#_all_cores_filled_ = False
	#while(_all_cores_filled_ == False):
		#centres, distributed = kmeans(state_space,K)
		#print("going into k means" + "we only have " + str(np.max(distributed)))
		#if np.max(distributed) == K-1:
			#_all_cores_filled_ = True
	
	# bhack to make just the K means work
	
	centres, stuff = kmeans(state_space,K)
	
	# hack ends here	
	
	#proportions = partition_algo_distances(state_space,centres,tau)
	proportions = partition_algo_distances_tight(state_space,centres,tau)
	sub_state_space, sub_prob = seperate_via_proportions(state_space,proportions, p)
	return sub_state_space, sub_prob, centres

'''
The following function will help seperate the state space with the given means.

Inputs: state_space : state_space = N \times D matrix, N number of states and D is the dimension of a state.
        p : N \times 1 vector which will have the probabilties of the corresponding states in state space.
	K : number of cluster we want to split it up into.
	tau is the number less then 1. Is the threshold after which the proportion is turned to zero. It is a ratio of the distances, so if the ratio of the distance to one mean is is less that tau then we change its proportion to zero.
Output:
	sub_state_spaces : ( M \times D ) \times K : List of the state space going out to each core.
	sub_probs :  ( M \times 1 ) \times K : List of the probabilities going out to each core.
'''
def seperate_via_Means(state_space,p,centres,K,tau=0.1):
	#proportions = partition_algo_distances(state_space,centres,tau) # weak paritioning promiting overlap
	proportions = partition_algo_distances_tight(state_space,centres,tau) # tight partioning no overlap
	sub_state_space, sub_prob = seperate_via_proportions(state_space,proportions, p)
	return sub_state_space, sub_prob


'''
The following function will help seperate the state space with the given means and is stingy.

Inputs: state_space : state_space = N \times D matrix, N number of states and D is the dimension of a state.
        p : N \times 1 vector which will have the probabilties of the corresponding states in state space.
	K : number of cluster we want to split it up into.
	tau is the number less then 1. Is the threshold after which the proportion is turned to zero. It is a ratio of the distances, so if the ratio of the distance to one mean is is less that tau then we change its proportion to zero.
Output:
	sub_state_spaces : ( M \times D ) \times K : List of the state space going out to each core.
	sub_probs :  ( M \times 1 ) \times K : List of the probabilities going out to each core.
'''
def seperate_via_Means_stingy(state_space,p,centres,K,rank,tau=0.1):
	#proportions = partition_algo_distances(state_space,centres,tau) # weak paritioning promiting overlap
	proportions = partition_algo_distances_tight_stingy(state_space,centres,tau,rank) # tight partioning no overlap
	sub_state_space, sub_prob = seperate_via_proportions(state_space,proportions, p)
	return sub_state_space, sub_prob

'''
The following function will concatenate the states and probabilities from all the cores.

Inputs: List[ State spaces ] : We need to get a list of all the state spaces
	List[ Probabilities ] : We will get a list of all the probabilties.
	Nils : This tells us the number of nils that are coming through.
	K : Number of cores.
	
Outputs: state_space_new : the new state space with all uniques states
	 p_new : the new probability vector with positions corresponding to the state_space_new.

Notes: The state space and probabilities have to come as numpy arrays.
'''
def Concatenate_state_space(L_state_space,L_prob,Nils,K):

	if Nils != K-1:
	
		# Here we delete the Nils in the system.
		for i in range(Nils):
			#print("REMOVED "+str(i) + " NILS ")
			L_state_space.remove("Nil")
			L_prob.remove("Nil")
		
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
	else:
		for i in range(K):
			if L_state_space[i] != 'Nil':
				state_space_new = L_state_space[i]
				p_new = L_prob[i]
				break
		return state_space_new, p_new
	
''' 
NAME: SIMPLE SEND AND RECEIVE ALL THE STATES AND PROBABILITIES.

We are now interested in setting up a network interface to distribute the data around to each of the cores. The first one will be a simple one which will have a k **2 complexity.

Inputs : Data , this is the data to be broadcasted to everyone.
	 rank , rank of the core
	 K, The number of cores in total to communicate. 
	 comm, The communication link of mpi4py
	 
Outputs: States, The list of all the states collected from each core.
	 probs, The list of all the states collected from each core.

'''
def simple_SR_all_states(sub_state_space,sub_prob,rank,K,comm):
	# Interesting
	states = []
	probs = []
	Nils = []
	Nil_val = 0
	for i in range(K):
		if i == rank-1:
			for j in range(K):
				if j != rank-1:
					# We need to send the right package.
					if sub_prob[j] == "Nil":
						Nil_val = 1
					comm.send([sub_state_space[j],sub_prob[j],Nil_val],dest = j+1)
				else:
					if sub_prob[j] == "Nil":
						Nil_val = 1
					states.append(sub_state_space[rank-1])
					probs.append(sub_prob[rank-1])
					Nils.append(Nil_val)
				Nil_val = 0
		else:
			temp = comm.recv(source=i+1)
			states.append(temp[0])
			probs.append(temp[1])
			Nils.append(temp[2])

			
	return states, probs, np.sum(Nils)

''' 
NAME: SIMPLE SEND AND RECEIVE ALL THE MEANS.

Inputs : Data , this is the data to be broadcasted to everyone.
	 rank , rank of the core
	 K, The number of cores in total to communicate. 
	 
Outputs: mu, The list of all the means collected from each core.

'''
def simple_SR_all_means(Data,SS_size,sum_prob,rank,D,K,comm):
	# Interesting
	mu = np.zeros((K,D))
	sizes = np.zeros((K,))
	cumm_probs = np.zeros((K,))
	for i in range(K):
		if i == rank-1:
			for j in range(K):
				if j != rank-1:
					comm.send([Data,SS_size,sum_prob],dest = j+1)
				else:
					mu[rank-1,:] = Data
					sizes[i] = SS_size
					cumm_probs[i] = sum_prob
		else:
			temp = comm.recv(source=i+1)
			mu[i,:] = temp[0]
			sizes[i] = temp[1]
			cumm_probs[i] = temp[2]
	return mu, sizes, cumm_probs
			
'''
NAME: COMPUTE THE MEAN OF A STATE SPACES WITH THE PROBABILTIES.

Inputs : States : the state space in row vectors.
	 P : as a probability vector.
Outputs: mu: expectation as a vector of dimension D.

'''
def expectation(states,p):
	weighted_states = states * p[np.newaxis, :]
	mu = np.add.reduce(weighted_states, axis=1)
	return mu
	
''' 
Name: Averaging

Inputs: States : the state space in row vectors.

Output: mu: average of the state space.

'''

def averaging(states):
	return np.sum(states,axis=0)/np.array([states.shape[0]]*states.shape[1]).astype(float)

'''
NAME: Compress

Inputs: fsp_solver: Fsp solver object
	domain_enum: The domain indexing function
	epsilon : The amount to compress by in the L**1 norm
	t: time at which this is happening.
Output:
	fsp_solver : New solver with the intial condition given by the new state space and probability vector.
'''

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
'''
Name: Buncker Routine 

Description: Since the cores here are moving, we will have cases where they will move so far that they loose most of their states to compression. Hence in this case we want the core to respaun in the middle of the core with the most amount of states, splitting the big sized core into two.

Inputs: t : time step doing into the bunker
	delta_t : the time steps it might spend inside the buncker
	h : number of time steps before we trade
	mu : the means of each of the core from the previous time step
	sizes: the sizes of the state space in the previous time step
	rank : rank of the core
	comm: the communication portal to deliever through
Outputs:
	sub_state_space : new subspace that it is leaving with.
	sub_prob : new probabilities
	Nils : Number of Nils
	t : time to update
'''

def Buncker(t,delta_t,h,mu,sizes,rank,comm,K,D):
	# First we need to do this untils Nils != K
	_nil_flag_ = True
	print("WAIITINGGGGGGGGGG IN BUNKERRRRRRRRRRR "+ str(rank))
	while _nil_flag_ == True:
		# now we need to pick a new mu and send it out.
		new_mu = np.maximum(mu[np.argmax(sizes),:] - np.random.randint(0,3,(D,)),0) # 3 here is the magic number
		
		# now we are ready to recieve new means
		mu, sizes = simple_SR_all_means(new_mu,0,rank,len(new_mu),K,comm)
		
		# Recieve new states 
		
		sub_state_space, sub_prob, Nils = simple_SR_all_states(['Nil']*K,['Nil']*K,rank,K,comm)

		t = t+h*delta_t
		
		if Nils != K:
			_nil_flag_ = False
			print("OUT OF BUNCKERRRRRRRRRRRRRRRRRRRRR "+ str(rank))
	return sub_state_space, sub_prob,Nils,t
		
'''
Computing the marginal distribution given the state space matrix and the probability vector
Inputs:
	state_space: The State space which is a N * D matrix. D is the dimension of the problem
	p : N * 1 vector of postive values below 1.
	D : Dimension of the problem.
	name: Name of file, it has to be a string.
Output:
	A file is saved with a name.
	return 0
'''
def plot_marginals(state_space,p,D,name,rank,t,to_file = False):
	import matplotlib
	#matplotlib.use("PDF")
	#matplotlib.rcParams['figure.figsize'] = 5,10
	import matplotlib.pyplot as pl
	pl.suptitle("time: "+ str(t)+" units")
	print("time : "+ str(t))
	for i in range(D):
		marg_X = np.unique(state_space[:,i])
		A = np.where(marg_X[:,np.newaxis] == state_space[:,i].T[np.newaxis,:],1,0)
		marg_p = np.dot(A,p)
		pl.subplot(int(D/2)+1,2,i+1)
		pl.plot(marg_X,marg_p)
		if to_file == True:
			string_state = str(t) 
			string_prob = str(t) 
			f = open("Visuals/cummulative_state_"+ str(i)+".txt",'a')
			g = open("Visuals/cummulative_prob_"+ str(i)+".txt",'a')
			for j in range(len(marg_X)):
				string_state +=','+ str(marg_X[j])
				string_prob += ','+ str(marg_p[j])
			string_state += '\n'
			string_prob += '\n'
			f.write(string_state)
			g.write(string_prob)
			f.close()
			g.close()	
	 

	pl.savefig("Visuals/marginal_"+name+".pdf",format='pdf')
	pl.show()
	pl.clf()

''' 
Function : Simple Compress

Inputs : State space
	 Probabilities
	 error
output : Compressed state space
	 Compressed Probs
'''

def simple_compress(state_space,p,eps):
	
	# we will sort it and add up accumulatively and truncate
	arg_sort_p = np.argsort(p)
	a_s_p_a = np.add.accumulate(p[arg_sort_p])
	remove_num = np.sum(np.where(a_s_p_a < eps,1,0))
	
	return state_space[arg_sort_p[remove_num:],:], p[arg_sort_p[remove_num:]]
	


'''
Function : Seperate Via Marginals

Description: We seperate the state space into small pieces using the mean and std of the marginal.

WARNING ONLY FOR SINGLE PEEK PROBABILTIES.

Inputs: State Space: N times D array
	p : N times 1 array
	K : Number cores.

Outputs: 

'''
def seperate_via_marginal(state_space,p,K,D):

	def mean(X,p):
		return np.sum(X*p)

	def std(X,p,mu):
		return np.sqrt( np.sum(np.power(X,2)*p) - mu**2)

	# The means to go out
	centres = np.zeros((K,D))
		
	# Compute the marginal
	
	for i in range(D):
		marg_X = np.unique(state_space[:,i])
		A = np.where(marg_X[:,np.newaxis] == state_space[:,i].T[np.newaxis,:],1,0)
		marg_p = np.dot(A,p)
		
		### Uniform partitioning method ###
		'''
		max_point = np.amax(marg_X)
		min_point = np.amin(marg_X)
		
		delta_X = (max_point - min_point)/K
		
		for k in range(K):
			centres[k,i] = delta_X*(k+1)
		'''
		
		### Cummulative Partitioning ###
		
		cumm_marg_p = np.cumsum(marg_p)
		
		delta_p = cumm_marg_p[-1]/K
		
		_counter_ = 0
		_pre_pos_ = 0
		
		for k in range(K):
			while( cumm_marg_p[_counter_] < delta_p*(k+1) ):
				_counter_ += 1
			centres[k,i] = marg_X[int((_counter_ + _pre_pos_))]
			_pre_pos_ = _counter_
			
		
		### Peeks method ###
		'''
		peek = np.amax(np.where(marg_X == int(mean(marg_X,marg_p)),range(len(marg_X)),0))
		num_peeks = 1 ##### CLEARLY ONLY FOR SINGLE PEEKS AT THE MOMENT
		
		num_pos = (K/num_peeks)/2.0
		
		mu = mean(state_space[:,i],p)
		
		#print("mean in the dimension "+ str(i) + "is : "+ str(mu))
		stdp = std(state_space[:,i],p,mu)
		#print("std in the dimension "+ str(i) + "is : "+ str(stdp))		
		for j in range(int(num_pos)):
			centres[2*j,i] = np.maximum(mu + (j+1)*stdp,0)
			centres[2*j+1,i]=np.maximum(mu- (j+1)*stdp,0)
		'''
		
	#print("THE centres ARE \n "+ str(centres))
	proportions = partition_algo_distances_tight(state_space,centres,0.01) # The Tau here is a dummy and it needs replacing.
	sub_state_space, sub_prob = seperate_via_proportions(state_space,proportions,p)
	return sub_state_space, sub_prob, centres

'''
Function: Seperate via the shortest dimension

'''
def seperate_via_shortest_dim(state_space,p,K,D):
	# find the shortest dimension.
	min_dim = 0
	temp_num = np.unique(state_space[:,0]).shape[0]
	for i in range(D-1):
		num_elements = np.unique(state_space[i+1,0]).shape[0]
		if  num_elements < temp_num:
			min_dim = i+1
			temp_num = num_elements
	# Now we have the smallest dimension.
	# We wish to now split evenly accross this.

'''
Function : Seperate Via Marginals

Description: We seperate the state space into small pieces using the mean and std of the marginal.

WARNING ONLY FOR SINGLE PEEK PROBABILTIES.

Inputs: State Space: N times D array
	p : N times 1 array
	K : Number cores.

Outputs: 

'''
def seperate_via_marginal_blocks(state_space,p,K):
	D = state_space.shape[1]
	N = state_space.shape[0]
	
	def mean(X,p):
		return np.sum(X*p)

	def std(X,p,mu):
		return np.sqrt( np.sum(np.power(X,2)*p) - mu**2)

	# The means to go out
	centres = np.zeros((K,D))
		
	# Compute the marginal
	
	for i in range(D):
		marg_X = np.unique(state_space[:,i])
		A = np.where(marg_X[:,np.newaxis] == state_space[:,i].T[np.newaxis,:],1,0)
		marg_p = np.dot(A,p)

		

		
		### Cummulative Partitioning ###
		
		cumm_marg_p = np.cumsum(marg_p)
		
		delta_p = cumm_marg_p[-1]/K
		
		_counter_ = 0
		
		for k in range(K-1):
			while( cumm_marg_p[_counter_] < delta_p*(k+1) ):
				_counter_ += 1
			centres[k,i] = marg_X[_counter_]
		centres[-1,i] = marg_X[-1]
		
			
		
	# We do the partitioning manuelly.
	proportions = np.zeros((N,K))
	
	previous_val = 0
	for k in range(K):
		temp = np.ones((N,))
		for i in range(D):
			# Here we need to find all the right terms
			temp = temp*np.where(state_space[:,i] <= centres[k,i],1,0)
		proportions[:,k] = temp
	
	print("core 0 has "+ str(np.sum(proportions[:,0])))
	for k in range(K-1):
		for j in range(k+1):
			proportions[:,k+1] -= proportions[:,j]
		print("core k has "+ str(np.sum(proportions[:,k])))
	# precautions:
	#print("we have "+ str(np.sum(proportions)) + " we should have "+ str(N))
	if np.sum(proportions) != N:
		print("FAILL")
			
	sub_state_space, sub_prob = seperate_via_proportions(state_space,proportions,p)
	return sub_state_space, sub_prob, centres

#def seperate_via_even_dist(state_space,p,K):
	#N = state_space.shape[0]
	#D = state_space.shape[1]
	
	#ordering = np.lexsort(state_space.T)
	
	#state_space = state_space[ordering,:]
	#p = p[ordering]
	
	## now we break it up into even pieces.
	#L_states = []
	#L_probs = []
	
	#for k in range(K):
		#L_states.append(state_space[k*(N/K):(k+1)*(N/K),:])
		#L_probs.append(p[k*(N/K):(k+1)*(N/K)])
		
	#return L_states,L_probs, 0
		
#def seperate_via_even_dist(state_space,p,K):
	

	## we are trying to balance the probability on each of the cores.
	#N = state_space.shape[0]
	#D = state_space.shape[1]
	
	#ordering = np.argsort(p)[::-1]
	
	##state_space = state_space[ordering,:]
	##p = p[ordering]
	### now we break it up into even pieces.
	#L_states = []
	#L_probs = []
	
	#for k in range(K):
		#L_states.append(state_space[ordering[k::K],:])
		#L_probs.append(p[ordering[k::K]])
		
	#return L_states,L_probs, 0


def seperate_via_even_spread(state_space,p,K):
	

	# we are trying to balance the probability on each of the cores.
	N = state_space.shape[0]
	D = state_space.shape[1]
	
	ordering = np.argsort(p)[::-1]
	
	max_threshold = np.sum(p)/float(K)
	
	balancers = ordering[p[ordering]>max_threshold]
	
	if len(balancers) < K:
		# frist divide the values
		append_p = []
		append_states = []
		for item in balancers:
			p[balancers] = p[balancers]/float(K)
			append_p.append(p[balancers])
			append_states.append(state_space[balancers,:])
		
		#second append the new states. K-1 of the same state.
		for i in range(len(balancers)):
			for k in range(K-1):
				p = np.concatenate((p,append_p[i]))
				state_space = np.concatenate((state_space,append_states[i]),axis=0)
		
		ordering = np.argsort(p)[::-1]
		# Then reorder.
	
	
	L_states = []
	L_probs = []
	
	for k in range(K):
		L_states.append(state_space[ordering[k::K],:])
		L_probs.append(p[ordering[k::K]])
		
	return L_states,L_probs, 0
	
	

def seperate_via_rectangles(state_space,p,K):
	
	import itertools
	small_d = int(np.log2(K))
	D = state_space.shape[1]
	N = state_space.shape[0]
	
	# first to find the order of cutting.
	
	Margs = []
	Num_Margs = []
	for i in range(D):
		Margs.append(np.unique(state_space[:,i]))
		Num_Margs.append(len(Margs[-1]))
	print(str(Num_Margs))
	order_to_split = np.arange(0,D,1)[np.argsort(Num_Margs)]
	print(str(order_to_split))
		
	# now for each dimension we need to split it into two and put them in two lists.
	A = []
	B = []
	
	for i in range(small_d):
		marg_X = np.unique(state_space[:,order_to_split[i]])
		elements = 0
		split = 1
		for x in marg_X:
			elements += np.sum(np.where(state_space[:,order_to_split[i]]== x,1,0))
			if elements > N/2.0:
					A.append(marg_X[0:split])
					B.append(marg_X[split:])
					break
			else:
				split += 1
	# now that we have the A and B to split, we do the next processing.
	vecs = []
	for i in range(small_d):
		vecs.append([])
		vecs[-1].append(np.sum(np.where(A[i][:,np.newaxis] == state_space[:,order_to_split[i]].T[np.newaxis,:],1,0),axis=0))
		vecs[-1].append(np.sum(np.where(B[i][:,np.newaxis] == state_space[:,order_to_split[i]].T[np.newaxis,:],1,0),axis=0))
	# we have all the vectors. Ready to go for splitting.
	
	permutes = itertools.product([0,1],repeat=small_d)
	# for each combinations, multiply the vectors and bundle them.
	sub_state_space = []
	sub_prob = []
	for perm in permutes:
		print("AH" + str(perm))
		temp_vec = np.ones(N)
		for i in range(small_d):
			temp_vec *=vecs[i][perm[i]] 
		index_keep = np.nonzero(temp_vec)
		#print("ahh"+str(index_keep[0].shape))
		sub_state_space.append(state_space[index_keep[0],:])
		sub_prob.append(p[index_keep[0],:])
	return sub_state_space, sub_prob, 0
		
		