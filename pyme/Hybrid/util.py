'''
Vikram Sunkara. Utility file containing some basic functions

'''
# imports
import numpy as np
import pdb
import itertools

def stochiometric_split(V, propensities, deter_vector):
  
	# deter_vector is N_s vector which is one is the variable is deterministic and zero if stochastic
	
	N_s = len(V[0])
	
	stoc_vector = np.where( deter_vector == False, True, False)
	
	print(" det_vec " + str(deter_vector))
	print(" Stoc_vec" + str(stoc_vector))
	
	# Need to loop through and seperate them
	
	V_d = []
	V_s = []
	
	#new_prop_d = []
	new_prop_s = []
	
	stoc_positions = np.zeros((len(V),),dtype=np.bool)
  
	for i in range(len(V)):
	
		# First V_d
		temp_s = V[i].copy()
		
		#print( str(temp_d) +" arg "  +str(temp_d[deter_vector] )) 
		
		temp_s[deter_vector] = 0
		
		#print( str(temp_d))
		
		if np.sum(np.abs(temp_s)) != 0:
			V_s.append(temp_s)
			new_prop_s.append(propensities[i])
			stoc_positions[i] = True
		else:
			stoc_positions[i] = False
		
		# Second V_s
		temp_d = V[i].copy()
		
		#print( str(temp_s) +" arg "  +str(temp_s[stoc_vector] )) 
		
		temp_d[stoc_vector] = 0
		
		#print( str(temp_s))
		
		if np.sum(np.abs(temp_d)) != 0:
			V_d.append(temp_d)
			#new_prop_d.append(propensities[i])
			
	return V_d, V_s, stoc_positions, new_prop_s
	
#def positions(X, V_s, stoc_vector):
  
	#''' Get the positions of the previous positions. '''
	
	## X is the state space vector. N \times N_s
	
	## stoc_vector is a vector $N_s$ with 1 when a variable is stochastic and zero otherwise.
	
	## Initialising the positions
	
	#N = len(X) # Number of states.
	
	#N_s = np.sum(stoc_vector)
	
	#N_r_s = len(V_s) # N_r_s is the number of propensities which are purely stochastic ( N_r_s = len(V_s))
	
	#position = np.zeros((N,N_r_s),dtype=np.int)
	#valid = np.zeros((N,N_r_s),dtype=np.bool)
	#shift_M = np.zeros((N_r_s,N,N_s),dtype=np.int)
	
	## Loops through the stochiometry and find the coresponding indexes.
	
	#for i in range(N_r_s):
	
		#shift = X - np.array(V_s[i])
		#X_small = X[:,stoc_vector].copy()
		#Shift_small = shift[:,stoc_vector].copy()

		#temp_mat = np.sum(np.abs((X_small[:,:,np.newaxis]-Shift_small.T[np.newaxis,:,:])),axis=1)==0
		
		#pos_mat =  np.where(temp_mat == True, 1,0)
		
		
		#valid[:,i] = np.sum(pos_mat,axis =0) > 0
		
		
		#position[:,i] = np.argmax(pos_mat,axis=0)
		#shift_M[i,:,:] = Shift_small[:]
		
	#return valid, position, shift_M
	
def positions(X, V_s, stoc_vector,domain_enum):
  
	''' Get the positions of the previous positions. '''
	
	# X is the state space vector. N \times N_s
	
	# stoc_vector is a vector $N_s$ with 1 when a variable is stochastic and zero otherwise.
	
	# Initialising the positions
	##pdb.set_trace()
	N = X.shape[1] # Number of states.
	
	N_s = np.sum(stoc_vector)
	
	N_r_s = len(V_s) # N_r_s is the number of propensities which are purely stochastic ( N_r_s = len(V_s))
	
	position = np.zeros((N,N_r_s),dtype=np.int64)
	valid = np.zeros((N,N_r_s),dtype=np.bool)
	#shift_M = np.zeros((N_r_s,N,N_s),dtype=np.int)
	
	# Loops through the stochiometry and find the coresponding indexes.
	##pdb.set_trace()
	for i in range(N_r_s):
		pre_states = X - np.array(V_s[i])[:,np.newaxis]
		interior = domain_enum.contains(pre_states)
		#print("shape In" + str(interior.shape))
		#print("shape valid" + str(valid[:,i].shape)) 
		valid[:,i] = interior
		#exterior = np.invert(interior)
		if np.sum(valid[:,i]) >0:
			position[interior,i] = domain_enum.indices(pre_states[:,interior])
		
	return valid, position
		
		
def Merge_all(List_SS_combined,list_p,list_Eta,K, N, D_s, D_d):
	
	SS_combined = np.zeros((N, D_s))
	P = np.zeros((N,))
	Eta = np.zeros((D_d,N))
	
	#Fill the matricies for quick sorting.
	start_position = 0
	
	for i in range(K):
		if List_SS_combined[i] != "Nil":
			end_position = start_position + List_SS_combined[i].shape[1]
			SS_combined[start_position:end_position,:] = List_SS_combined[i].T
			P[start_position:end_position] = list_p[i][:]
			Eta[:,start_position:end_position] = list_Eta[i][:]
			start_position = end_position
	
	#N = SS_combined.shape[0]
	# lexisort
	indexes =  np.lexsort(SS_combined.T)
	maps = indexes.copy()
	Sorted_SS_C = SS_combined[indexes,:]
	P = P[indexes]
	Eta = Eta[:,indexes]
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
		from cmepy import lexarrayset
		from cmepy.cme_matrix import non_neg_states as validity_test
		import itertools
		#from util import Concatenate_state_space as CSS # takes list of matricies.
		#from util import simple_compress as SC # return the compressed vectors and states.
		import time
		
		nabla = kwargs['domain_states']  # state space is D \times N
		return_domain = kwargs['domain_states']
		
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
				temp_w_terms = map(self.propensities[temp_i],* nabla[:,valid][:,new_states_index])*u[valid][new_states_index]
				positive_terms = temp_w_terms > 0
				temp_w.append(temp_w_terms[positive_terms])
				temp_new_states.append(temp_states[self.stoc_vector,:][:,valid][:,new_states_index][:,positive_terms]) # new stochastic states


				
				temp_cond_exp.append(np.multiply(temp_pre_states[:,positive_terms ]+np.array(transition)[deter_vector][:,np.newaxis],
temp_w[-1][np.newaxis,:])) # with transition

				#temp_cond_exp.append(np.multiply(nabla[deter_vector,:][:,new_states_index],temp_w[-1][np.newaxis,:]))
			
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

class Hybrid_HL_N_step_expander(object):
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
		from cmepy import lexarrayset
		from cmepy.cme_matrix import non_neg_states as validity_test
		import itertools
		#from util import Concatenate_state_space as CSS # takes list of matricies.
		#from util import simple_compress as SC # return the compressed vectors and states.
		import time
		
		nabla = kwargs['domain_states']  # state space is D \times N
		return_domain = kwargs['domain_states']
		
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
				temp_w_terms = map(self.propensities[temp_i],* nabla[:,valid][:,new_states_index])*u[valid][new_states_index]
				positive_terms = temp_w_terms > 0
				temp_w.append(temp_w_terms[positive_terms])
				temp_new_states.append(temp_states[self.stoc_vector,:][:,valid][:,new_states_index][:,positive_terms]) # new stochastic states


				
				temp_cond_exp.append(np.multiply(temp_pre_states[:,positive_terms ]+np.array(transition)[deter_vector][:,np.newaxis],
temp_w[-1][np.newaxis,:])) # with transition

				#temp_cond_exp.append(np.multiply(nabla[deter_vector,:][:,new_states_index],temp_w[-1][np.newaxis,:]))
			
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
		out_states[deter_vector,:] = nabla[deter_vector,0][:,np.newaxis]
		return out_states, out_states, out_u


		
def derivative_G(propensities,V,X,w,deter_vector,stoc_positions, positions, valid):
	
	# just the deterministics
	X_d = X[deter_vector,:].copy()
	temp_eta = np.zeros((np.sum(deter_vector),X.shape[1]))
	j = 0
	for i in range(len(stoc_positions)):
		##pdb.set_trace()
		# If x-\nu_i is non zero
		if stoc_positions[i] == True:
			
			if np.sum(valid[:,j]) != 0:
				#print(" X shape: " + str(X.shape))
				#print(" w shape: " + str(w.shape))
				#print("test :" + str(map(propensities[i],*X[:,positions[valid[:,j]][:,j]])))
				
				
				temp_eta[:,valid[:,j]] += (X_d[:,positions[valid[:,j]][:,j]] 
							    - X_d[:,valid[:,j]] +
								V[i][deter_vector][:,np.newaxis]
							  )*map(propensities[i],* X[:,positions[valid[:,j]][:,j]])*w[positions[valid[:,j]][:,j]]
			j += 1
		else:
			temp_eta[:,:] += (V[i][deter_vector][:,np.newaxis])*map(propensities[i],* X)*w
			
	return_X = np.zeros(X.shape)
	return_X[deter_vector,:] = temp_eta
	return_X[np.invert(deter_vector),:] = X[np.invert(deter_vector),:].copy()
	return return_X
	#return temp_eta
	
def simple_FSP(model, X, w, h,stoc_vector ,expander ,eps_per_step):
  
	import cmepy.fsp.solver
	#import cmepy.fsp.simple_expander
	import cmepy.domain
	import cmepy.state_enum
	
	##pdb.set_trace()
	
	#order = np.lexsort(X)
	
	initial_state_enum = cmepy.state_enum.StateEnum(X,stoc_vector)

	fsp_solver=cmepy.fsp.solver.create(
		model,
		initial_state_enum.ordered_full_dom,
		initial_state_enum,
		expander,
		p_0 = w[initial_state_enum.index]
		)
	fsp_solver.step(h,eps_per_step)
	
	
	# we will test the output of the solver.
	#from scipy.integrate import ode
	
	###pdb.set_trace()
	
	#A, old_dom_enum = compute_A(fsp_solver.domain_states, model.propensities, model.transitions, stoc_vector)
	
	###pdb.set_trace()
	
	#new_p_0 = np.zeros((fsp_solver.domain_states.shape[1]+1,))
	
	#new_indexes = old_dom_enum.indices(X)
	
	#new_p_0[new_indexes] = w
	
	#def f(t,y,A):
		#return A.dot(y)
	
	#ode_obj = ode(f).set_integrator('vode',method='bdf')
	#ode_obj.set_initial_value(new_p_0, 0).set_f_params(A)
	
	###pdb.set_trace()
	#while ode_obj.successful() and ode_obj.t < h:
		#ode_obj.integrate(h)
	
	#print(str(ode_obj.y))
	
	##print(str(old_dom_enum.ordered_states))
	
	##print(str(old_dom_enum.ordered_full_dom))
		
	return  fsp_solver.domain_states, fsp_solver.y[0], fsp_solver.domain_enum, fsp_solver.y[1]



def implicit_euler(propensities,V,X,w,h,deter_vector,stoc_positions, positions, valid,deriv):
	
	#from util import derivative_G
	# Prefixed erroe tolrence.
	max_num_iterations = 100
	Tol = 1e-8
	
	stoc_vector = np.where(deter_vector == True, False, True)
	current_X = X.copy()
	next_X = np.zeros(current_X.shape)
	adjusted_X = X.copy()
	
	for n in range(max_num_iterations):
		##pdb.set_trace()
		
		# averaging the interation
		
		adjusted_X[deter_vector,:] = (current_X[deter_vector,:] + X[deter_vector,:])/2.0
		
		
		
		temp_der = deriv(propensities,V,adjusted_X,w,deter_vector,stoc_positions, positions, valid)
		
		next_X[deter_vector,:] = X[deter_vector,:] + h*np.divide(temp_der[deter_vector,:],w[np.newaxis,:])
		###pdb.set_trace()
		##print( " X[:,1] " + str(X[:,1]))
		##print( " Adjusted_X[deter,1] " + str(adjusted_X[:,1]))
		##print( " X[deter,1] " + str(X[deter_vector,1]))
		#print( "biggest change : " + str(np.abs(h*np.amax(np.divide(temp_der[deter_vector,:],w[np.newaxis,:])))))
		#print( " next_X[deter,1] " + str(next_X[deter_vector,1]))
		#print( " temp_derv[deter,1] " + str(temp_der[deter_vector,1]))
		##print( " Current_X[deter,1] " + str(current_X[deter_vector,1]))
		#print( " difference " + str(np.sum(np.power(next_X[deter_vector,:] - current_X[deter_vector,:],2))))
		if  np.sum(np.power(next_X[deter_vector,:] - current_X[deter_vector,:],2)) < Tol : # L_2 norm test
			next_X[stoc_vector,:] = X[stoc_vector,:]
			#print("exiting on iteration : " + str(n))
			return next_X
		else:
			current_X[deter_vector,:] = next_X[deter_vector,:].copy()
		
		if n == max_num_iterations - 2:
			print(" Did not converge ")
			print(" Difference : " + str(np.sum(np.power(next_X[deter_vector,:] - current_X[deter_vector,:],2))))
			
	next_X[stoc_vector,:] = X[stoc_vector,:]
	
	return next_X
	
	## test the ode solver
	
	#def f(t,y,args):
		## args[1] = propensities
		## args[2] = V
		## args[3] = X
		## args[4] = deter_vector
		## args[5] = stoc_positions
		## args[6] = positions
		## args[7] = valid
		## arg[8] = w
		#args[3][args[4],:] = y[:]
		#return np.divide(deriv(args[1],args[2],args[3],args[8],args[4],args[5], args[6], args[7]),args[8])
	
	#from scipy.integrate import ode
	###pdb.set_trace()
	#deter_ode = ode(f).set_integrator('vode',method='adams')
	#deter_ode.set_initial_value(X[deter_vector,:], 0).set_f_params([propensities,V,X,w,deter_vector,stoc_positions, positions, valid])
	
	###pdb.set_trace()
	#while deter_ode.successful() and deter_ode.t < h:
		#deter_ode.integrate(h)
	
	#print("Black Box: \n"+ str(deter_ode.y))
	
	##print("iterator : \n:"+str(next_X[deter_vector,:]))
	#X[deter_vector,:] = deter_ode.y
	 
	#return X

def compute_A(states, propensities, transitions, stoc_vector):
	
	import cmepy.state_enum
	import numpy
	
	domain_enum = cmepy.state_enum.StateEnum(states,stoc_vector)
	reactions = itertools.izip(propensities, transitions)

	src_states = numpy.array(domain_enum.ordered_full_dom)
	src_indices = domain_enum.indices(src_states)

	A = np.zeros((states.shape[1]+1,states.shape[1]+1))
	

	deter_vector = numpy.invert(stoc_vector)
	 
	for (propensity, transition) in reactions:
		# compute destination states for this transition
		transition = numpy.asarray(transition)[:, numpy.newaxis]
		dst_states = src_states + transition
        
		# determine which states have destination states inside the
		# truncated domain. these will be defined as the 'interior' states.
		# conversely, 'exterior' states are those states of the truncated
		# domain with destination states not in the domain.
		##pdb.set_trace() 
		interior = domain_enum.contains(dst_states)
		exterior = numpy.logical_not(interior)

		num_int_states = numpy.add.reduce(interior)
		num_ext_states = numpy.add.reduce(exterior)

		# these lists will be used to accumulate 'COO'-ordinate format
		# sparse matrix data for this reaction.

		#data = []
		#rows = []
		#cols = []

		# account for the sparse matrix data for the flux out of the
		# interior states of the truncated domain

		if num_int_states > 0:
			int_src_states = numpy.array(src_states[:, interior])
			int_src_indices = numpy.array(src_indices[interior])
			int_dst_states = numpy.array(dst_states[:, interior])
			int_dst_indices = domain_enum.indices(int_dst_states)
			int_coefficients = np.array(map(propensity,* int_src_states))            

			## flux out
			#data.append(-int_coefficients)
			#cols.append(int_src_indices)
			#rows.append(int_src_indices)
			A[int_src_indices,int_src_indices] -= int_coefficients
			## flux in
			#data.append(int_coefficients)
			#cols.append(int_src_indices)
			#rows.append(int_dst_indices)
			A[int_dst_indices,int_src_indices] += int_coefficients
			
	A[-1,:] = np.sum(A,axis=1)
	return A, domain_enum
## Testing functions:
#X = np.zeros((4,3))
#X[0,:] = [0,2,0]
#X[1,:] = [1,2.1,1]
#X[2,:] = [0,2.2,1]
#X[3,:] = [1,2.3,0]

#p = np.zeros((4,))
#p[0] = 0.2
#p[1] = 0.3
#p[2] = 0.16
#p[3] = 0.28

#V= []
#V.append(np.array((-1,1,0)))
#V.append(np.array((0,-1,1)))
#V.append(np.array((1,0,-1)))
#V.append(np.array((1,0,0)))

#propensities = []
#propensities.append(lambda *x : x[0])
#propensities.append(lambda *x : x[1])
#propensities.append(lambda *x : x[2])
#propensities.append(lambda *x : 10.0)

#deter_vector = np.array((False,True,False))
#stoc_vector = np.array((True,False,True))

#V_d, V_s, stoc_positions, new_prop_s = stochiometric_split(V, propensities, deter_vector)

#print(" V_d : " + str(V_d))
#print(" V_s : " + str(V_s))
#print(" V_s Position:" + str(stoc_positions))

#valid, position, shift_M = positions(X, V_s, stoc_vector)

#print("valid: \n " + str(valid))
#print("position\n " + str(position))

#print(" Valid :\n " + str(vaild) )
#print(" position\n : " + str(position) )
#print(" shift_M \n : " + str(shift_M) )

#for i in range( vaild.shape[1] ):
#	#print(" valid " + str(vaild[:,i]))
#	print("X \n"+ str(X[vaild[:,i],:]) + "\n pos \n " + str(X[position[vaild[:,i],i],:]))




#h = 0.005

##new_X = derivative_G(propensities,V,X.T,p,deter_vector,stoc_positions, position, valid)
##print("new_X :\n" + str(new_X))
##forwarded = implicit_euler(propensities,V,X.T,p,h,deter_vector,stoc_positions, position, valid, derivative_G)
##print("forward :\n" +str(forwarded))
#import cmepy.fsp.solver

#model = cmepy.model.create(
	#propensities = new_prop_s,
	#transitions= V_s,
	#shape= (1,2,1),
	#initial_state= (0,3,0)
	#)
	
#test_expander = Hybrid_N_step_expander(model,stoc_vector,V, h, 0.005)
#new_domain, new_domain, u = test_expander.expand(domain_states = X.T, p = p)
#print(" X \n" + str(X.T))
#print( " new domain \n" + str(new_domain.T) + '\n new u\n' + str(u))



#X, w = simple_FSP(model, X.T, p, h,stoc_vector , test_expander ,0.005)

#print(" new X \n" + str(X.T))
#print(" new w \n" + str(w))

def compute_marginal(d,SS,p):
    # d is the dimension to extract.
    
    uniques = np.unique(SS[d,:])
    X = np.arange(0,int(np.amax(uniques))+1,1)
    new_p = np.zeros((len(X),))
    for i in range(len(p)):
        new_p[int(SS[d,i])] += p[i]
        
    return X, new_p