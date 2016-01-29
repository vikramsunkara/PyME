import numpy as np



##Computing the marginal distribution given the state space matrix and the probability vector
#@param state_space: The State space which is a (WARNING:) N * D matrix. D is the dimension of the problem
#@param p  N * 1 vector of postive values below 1.
#@param Name of file, it has to be a string.
#@param t time point of the data
#@param labels the labels you want to add to the subgraphs
def plot_marginals(state_space,p,name,t,labels = False,interactive = False):
	import matplotlib

	import matplotlib.pyplot as pl
	if interactive == True:	
		pl.ion()
	pl.clf()
	pl.suptitle("time: "+ str(t)+" units")
	#print("time : "+ str(t))
	D = state_space.shape[1]

	for i in range(D):
		marg_X = np.unique(state_space[:,i])
		A = np.where(marg_X[:,np.newaxis] == state_space[:,i].T[np.newaxis,:],1,0)
		marg_p = np.dot(A,p)
		pl.subplot(int(D/2)+1,2,i+1)
		pl.plot(marg_X,marg_p)
		pl.yticks(np.linspace(np.amin(marg_p), np.amax(marg_p), num=3))
		pl.axvline(np.sum(marg_X*marg_p),color= 'r')
		pl.axvline(marg_X[np.argmax(marg_p)],color='g')
		if labels == False:
			pl.xlabel("Specie: " + str(i+1))
		else:
			pl.xlabel(labels[i])
	if interactive == True:
		pl.draw()
	else:
		pl.tight_layout()
		pl.show()
	
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

def plot_2D_contour(states,p,labels):
	import pylab as pl
	
	from cmepy.statistics import expectation as EXP
	exp = EXP((states,p)) 
	X = np.unique(states[0,:])
	Y = np.unique(states[1,:])
	X_len = len(X)
	Y_len = len(Y)
	Z = np.zeros((X.max()+1,Y.max()+1))
	for i in range(len(p)):
		Z[states[0,i],states[1,i]] = p[i]
	pl.clf()
	XX, YY = np.meshgrid(X,Y)	
	pl.contour(range(X.max()+1),range(Y.max()+1),Z.T)
	pl.axhline(y=exp[1])
	pl.axvline(x=exp[0])
	pl.xlabel(labels[0])
	pl.ylabel(labels[1])
	pl.draw()
