# THe following COde is for Hallender Lodstet Model.

import numpy as np

def derivative_G(propensities,V,X,w,deter_vector,stoc_positions, positions, valid):
	
	# just the deterministics
	X_d = X[deter_vector,:].copy()
	temp_eta = np.zeros((np.sum(deter_vector),X.shape[1]))
	j = 0
	for i in range(len(stoc_positions)):
		temp_sum = np.sum(map(propensities[i],* X)*w)
		temp_eta[:,:] += (V[i][deter_vector][:,np.newaxis])*temp_sum
		
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
		#print(len(args))
		#print(str(y.shape))
		#print(str(args[2].shape))
		args[2][args[3],:] = y[:].reshape((np.sum(args[3]),args[2].shape[1]))
		
		return derivative_G(args[0],args[1],args[2],args[7],args[3],args[4], args[5], args[6]).flatten()

def implicit_black_box(propensities,V,X,w,h,deter_vector,stoc_positions, positions, valid,deriv):
	
	from scipy.integrate import ode
	#pdb.set_trace()
	deter_ode = ode(f).set_integrator('lsoda',method='bdf', with_jacobian=False)
	deter_ode.set_initial_value(X[deter_vector,:].flatten(), 0).set_f_params([propensities,V,X,deter_vector,stoc_positions, positions, valid,w])

	#pdb.set_trace()
	while deter_ode.successful() and deter_ode.t < h:
		deter_ode.integrate(h)

	#print("Black Box: \n"+ str(deter_ode.y))

	#print("iterator : \n:"+str(next_X[deter_vector,:]))
	X[deter_vector,:] = deter_ode.y.reshape(( np.sum(deter_vector),X.shape[1]))
		
	return X
	
	