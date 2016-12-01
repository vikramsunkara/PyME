import numpy as np
import pdb
import time

def derivative_G(propensities,V,X,w,deter_vector,stoc_positions, positions, valid,jac):
	
	# just the deterministics
	X_d = X[deter_vector,:].copy()
	temp_eta = np.zeros((np.sum(deter_vector),X.shape[1]))
	j = 0
	for i in range(len(stoc_positions)):
		# If x-\nu_i is non zero
		if stoc_positions[i] == True:
			
			if np.sum(valid[:,j]) != 0:
				#print(" X shape: " + str(X.shape))
				#print(" w shape: " + str(w.shape))
				#print("test :" + str(map(propensities[i],*X[:,positions[valid[:,j]][:,j]])))
				
				# original Terms
				temp_eta[:,valid[:,j]] += (X_d[:,positions[valid[:,j]][:,j]] 
							    - X_d[:,valid[:,j]] +
								V[i][deter_vector][:,np.newaxis]
							  )*map(propensities[i],* X[:,positions[valid[:,j]][:,j]])*w[positions[valid[:,j]][:,j]]
				
				# Correction terms
				# x terms
				temp_eta[:,:] -= jac(X,deter_vector,i)*w[np.newaxis,:] # these should be all the terms which are minusing out.
				# x-v_j term.
				temp_eta[:,valid[:,j]] += jac(X[:,positions[valid[:,j]][:,j]],deter_vector,i)*w[positions[valid[:,j]][:,j]][np.newaxis,:]
				
				
				
			j += 1
		else:
			temp_eta[:,:] += (V[i][deter_vector][:,np.newaxis])*map(propensities[i],* X)*w
			
	#return_X = np.zeros(X.shape)
	#return_X[deter_vector,:] = temp_eta
	#return_X[np.invert(deter_vector),:] = X[np.invert(deter_vector),:].copy()
	return temp_eta


def f(t,y,args):
		# args[1] = propensities
		# args[2] = V
		# args[3] = X
		# args[4] = deter_vector
		# args[5] = stoc_positions
		# args[6] = positions
		# args[7] = valid
		# arg[8] = w
		# arg[9] = jac
		#print(len(args))
		#print(str(y.shape))
		#print(str(args[2].shape))
		args[2][args[3],:] = y[:].reshape((np.sum(args[3]),args[2].shape[1]))
		
		return np.divide(derivative_G(args[0],args[1],args[2],args[7],args[3],args[4], args[5], 
args[6],args[8]),args[7][np.newaxis,:]).flatten()


def f_dummy(t,y,args):
	args[2][args[3],:] = y[:].reshape((np.sum(args[3]),args[2].shape[1]))
	return np.divide(derivative_G(args[0],args[1],args[2],args[7],args[3],args[4], args[5], args[6],args[8]),args[7][np.newaxis,:])

def implicit_black_box(propensities,V,X,w,h,deter_vector,stoc_positions, positions, valid,jac):
	
	# Adjustment for systems reaching steady state
	"""
	temp = derivative_G(propensities,V,X,w,deter_vector,stoc_positions,positions,valid,jac)
	#pdb.set_trace()
	valid_adjust_pos = np.where(np.sum(np.abs(temp),axis=0) < 1e-10,True,False)

	valid_adjust = valid[:,:]
	valid_adjust[valid_adjust_pos,:] = False

	print(" Reached Steady State %d"%(np.sum(valid_adjust_pos)))
	"""
	from scipy.integrate import ode
	#pdb.set_trace()
	deter_ode = ode(f).set_integrator('vode',method='bdf', with_jacobian=False)
	deter_ode.set_initial_value(X[deter_vector,:].flatten(), 0)
	#deter_ode.set_f_params([propensities,V,X,deter_vector,stoc_positions, positions,valid_adjust,w,jac])
	deter_ode.set_f_params([propensities,V,X,deter_vector,stoc_positions, positions,valid,w,jac])
	
	#pdb.set_trace()
	while deter_ode.successful() and deter_ode.t < h:
		deter_ode.integrate(h)
	#print("Black Box: \n"+ str(deter_ode.y))

	#print("iterator : \n:"+str(next_X[deter_vector,:]))
	X[deter_vector,:] = deter_ode.y.reshape(( np.sum(deter_vector),X.shape[1]))
	

	# Another adjust to compensate for non negative
	X = np.where(X < 0.0,0.0,X)

	return X
	
	