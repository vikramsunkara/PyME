import numpy as np
import state_enum
import scipy.sparse
import pdb

def optimise_csr_matrix(csr_matrix):
    """    
    Performs **in place** operations to optimise csr matrix data. Returns None.
    """
    # xxx todo profile performance using permutations / subsets of these
    csr_matrix.sum_duplicates()
    csr_matrix.eliminate_zeros()

def compute_A(states, propensities, positions, valid):
	# computes the A matrix given the indexing map.
	#A = scipy.sparse.csr_matrix((states.shape[1],states.shape[1]))
	
	matrix_shape =  (states.shape[1], )*2
	
	row = []
	col = []
	data = []
	
	for j in range(len(propensities)):
		if np.sum(valid[:,j]) > 0:
			
			# Dirchlet Boundary condition
			
			# we the propensity value for all states.
			int_coefficients = np.array(map(propensities[j],* states))
			
			int_src_indices = positions[valid[:,j],j]
			int_dst_indices = np.arange(0,states.shape[1],1)[valid[:,j]]
			
			#in_row.append(int_dst_indices)
			#in_col.append(int_src_indices)
			
			row.append(int_dst_indices)
			col.append(int_src_indices)
			
			
			# Inflow in the domain
			#inflow.append(int_coefficients[int_src_indices])
			data.append(int_coefficients[int_src_indices])
			
			# Outflow from the domain
			#out_row_col.append(np.array(range(states.shape[1])))
			#outflow.append(-int_coefficients)
			
			row.append(np.arange(0,states.shape[1],1))
			col.append(np.arange(0,states.shape[1],1))
			data.append(-int_coefficients)
			
	#pdb.set_trace()
	data = np.concatenate(data)
	cols = np.concatenate(col)
	rows = np.concatenate(row)

	# create coo matrix
	coo_data = (data, (rows, cols))
	reaction_matrix = scipy.sparse.coo_matrix(coo_data, matrix_shape)

	# convert to sparse csr format, then compress & optimise the storage
	reaction_matrix = reaction_matrix.tocsr()
	optimise_csr_matrix(reaction_matrix)
	
	return reaction_matrix