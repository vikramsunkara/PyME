import numpy as np
import pdb
# this is a simple N-term approximation.

def simple_compress(state_space,p,eps):
	
	#pdb.set_trace()
	# we will sort it and add up accumulatively and truncate
	arg_sort_p = np.argsort(p)
	a_s_p_a = np.add.accumulate(p[arg_sort_p])
	remove_num = np.sum(np.where(a_s_p_a < eps,1,0))
	
	## debuging
	#pdb.set_trace()
	#if remove_num > 0 :
		#print( "removing " + str(state_space[2,arg_sort_p[:remove_num]]))
	
	
	return state_space[:,arg_sort_p[remove_num:]], p[arg_sort_p[remove_num:]]

def compress_by_marginal(state_space,p,eps):
	
	# hard code to compress the mRNA dimension only.
	highest_RNA = int(np.amax(state_space[2,:]))
	
	removes = []
	prob_left = eps
	
	for i in range(highest_RNA):
		# go back wards and see how many one can remove.
		check_terms = state_space[2,:] == (highest_RNA - i)
		total_prob = np.sum(p[check_terms])
		if total_prob < prob_left:
			removes.append(check_terms)
			prob_left -= total_prob
		else:
			# not enough probability we should leave
			break
	if len(removes) > 0:
		remove = removes[0]
		for i in range(len(removes)-1):
			remove = remove + removes[i+1]
		print("Removed states : " + str(np.sum(remove)))
		return state_space[:,remove == 0], p[remove==0]
	else:
		print("Removed no states")
		return state_space,p