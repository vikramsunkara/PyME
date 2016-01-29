# The objective is to extract the marginal and conditional expectation for high dimensional distributions.
# Author: Vikram Sunkara
# Date: 15 Mar 2013

import numpy as np
import pdb
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#import matplotlib.pyplot as plt
import pickle
import pylab as pl

def load_initial_dist(Exact_Name,stoc_vector):
	# load the initial distribution.
	instates = open(Exact_Name+ '_states_.pck','rb')
	mid_point_domain_states = pickle.load(instates)
	instates.close()
	inp = open(Exact_Name+ '_prob_.pck','rb')
	mid_point_y = pickle.load(inp)
	inp.close()
	non_negs = (mid_point_y > 0)
	initial_states , p_dense = project_to_Hybrid(mid_point_domain_states.T[:,non_negs],mid_point_y[non_negs],stoc_vector)
	return initial_states, p_dense

def project_to_Hybrid(X,w,stoc_vector):
	
	Unique_X = []
	Y_for_X = []
	P_1 = []
	P_for_P_2 = []
	
	deter_vec = np.invert(stoc_vector)
	
	order = np.lexsort(X[stoc_vector,:])
	new_X = X[:,order]
	new_w = w[order]
	Unique_X.append(new_X[:,0][stoc_vector])
	Y_for_X.append([new_X[:,0][deter_vec]])
	P_1.append(new_w[0])
	P_for_P_2.append([new_w[0]])
	
	N = X.shape[1] -1
	#pdb.set_trace()
	for i in range(N):
		# Check if the states are the same
		if np.sum(np.abs(Unique_X[-1] - new_X[:,i+1][stoc_vector])) == 0:
			
			# We need to update the same Y and P_1
			Y_for_X[-1].append(new_X[:,i+1][deter_vec])
			P_for_P_2[-1].append(new_w[i+1])
			P_1[-1] += new_w[i+1]
			
		else:
			# update and make new terms.
			Unique_X.append(new_X[:,i+1][stoc_vector])
			Y_for_X.append([new_X[:,i+1][deter_vec]])
			P_1.append(new_w[i+1])
			P_for_P_2.append([new_w[i+1]])
			
	#pdb.set_trace()
	new_N = len(Unique_X)
	p_1 = np.array(P_1)
	output_X = np.zeros((X.shape[0],new_N))
	
	for i,x in enumerate(Unique_X):
		#pdb.set_trace()
		output_X[stoc_vector,i] = x
		#p_1[i] = P_1[i]
		if len(Y_for_X[i]) > 1:
			for j,y in enumerate(Y_for_X[i]):
				output_X[deter_vec,i] += y*P_for_P_2[i][j]
			#print(str(output_X[deter_vec,i]) + " and " + str(p_1[i]))
			output_X[deter_vec,i] = output_X[deter_vec,i]/p_1[i]
		else:
			output_X[deter_vec,i] = Y_for_X[i][0]
			
	return output_X, p_1
	
def plot_reg_2D_stoc(X,stoc_vector):
	
	deter_vec = np.invert(stoc_vector)
	
	dom_max = np.amax(X[stoc_vector,:]) + 1
	
	A = np.zeros((dom_max,dom_max))
	
	stoc_indexs = np.arange(0,X.shape[0],1)[stoc_vector].astype(int)
		
	for i,deter_element in enumerate(deter_vec):
		if deter_element == True:
			A[X[int(stoc_indexs[0]),:].astype(int), X[int(stoc_indexs[1]),:].astype(int)] = X[i,:]
			pl.figure(i)
			#ax = fig.gca(projection='3d')
			#surf = ax.plot_surface(X[int(stoc_indexs[0]),:].astype(int), X[int(stoc_indexs[1]),:].astype(int),X[i,:], rstride=1, cstride=1,
#cmap=cm.coolwarm,linewidth=0, antialiased=False)
			pl.contour(A,X[i,:])
			#ax.zaxis.set_major_locator(LinearLocator(10))
			#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
			#fig.colorbar(surf, shrink=0.5, aspect=5)
			pl.show()
			
def plot_marginals(state_space,p,D,name,rank,t,stoc_vector,to_file = False):
	import matplotlib
	#matplotlib.use("PDF")
	#matplotlib.rcParams['figure.figsize'] = 5,10
	import matplotlib.pyplot as pl
	pl.suptitle("time: "+ str(t)+" units")
	print("time : "+ str(t))
	for i in range(D):
		
		#if stoc_vector[i] == True:
		if True:
			marg_X, pos = np.unique(state_space[:,i],return_inverse=True)
			marg_p = np.zeros((len(marg_X),))
			#marg_p[pos] += p
			#pdb.set_trace()
			for k in range(len(state_space[:,i])):
				marg_p[pos[k]] += p[k] 
			#A = np.where(marg_X[:,np.newaxis] == state_space[:,i].T[np.newaxis,:],1,0)
			#marg_p = np.dot(A,p)
			pl.subplot(int(D/2)+1,2,i+1)
			pl.plot(marg_X.flatten(),marg_p.flatten())
			
		#else:
			#pl.subplot(int(D/2)+1,2,i+1)
			
			#order = np.argsort(p[::-1])
			#pl.plot(np.cumsum(p[order]),state_space[order,i])
		
		
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
	 

	#pl.savefig("Visuals/marginal_"+name+".pdf",format='pdf')
	pl.show()
	pl.clf()
	
#from scipy.stats import poisson

#X_len = 10

#initial_states = np.zeros((2,X_len*48),dtype=np.int)
#p_dense = np.zeros((X_len*48,))

#for i in range(X_len):
	#for j in range(48):
		#initial_states[:,i + X_len*j] = [i,j]
		#p_dense[i+X_len*j] = poisson.pmf(i,4)*poisson.pmf(j,20)
		
#stoc_vector = np.array((True,False))
#X, w = project_to_Hybrid(initial_states,p_dense,stoc_vector)

#plot_marginals(X.T,w,2,"test",0,0,False)

#print(" X " + str(X))
#print(" w " + str(w))