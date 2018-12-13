########################################################
# Written by : Valliappan                              #
# This is the formulation for the compact loss         #
# Reference paper https://arxiv.org/pdf/1801.05365.pdf #
# This code involves much lesser looping statement 	   #		
# compared to the author's implementation              #
########################################################
import numpy as np

def forward(X):
	#####################################
	###### input 				   ######
	###### X - dim (n x k)         ######
	###### n - batch size          ######
	###### Output 				   ######
	###### l_C - (1 x 1)           ######
	#####################################
	n,k = X.shape
	loss_c = 0
	batch_list = set([i for i in range(n)])

	for it in range(n):
		# print(list(batch_list-set([it])))
		j = np.array(list(batch_list - set([it]))) # all values of j != it
		m_i = np.sum(X[j],axis=0)/float(n-1)
		x_i = X[it]
		Z_i = m_i - x_i
		loss_c += np.sum(np.multiply(Z_i,Z_i))/float(n*k)

	return loss_c

def backward(X, iterative = False):

	#####################################
	###### input 				   ######
	###### X - dim (n x k)         ######
	###### n - batch size          ######
	###### Output 				   ######
	###### l_C - (1 x 1)           ######
	#####################################

	n,k = X.shape
	batch_list = set([i for i in range(n)])

	dx = np.zeros_like(X, dtype=np.float32);	# every X_ij has a derivative back

	for it in range(n):
		j = np.array(list(batch_list - set([it]))) # all values of j != it
		m_i = np.sum(X[j],axis=0)/float(n-1)
		x_i = X[it]
		z_i = m_i - x_i

		cum_sum = np.sum(z_i)
		const_term = 2/((n-1)*n*k)
		# with the for loop for 'j' 
		if(iterative == True):
			for j in range(0,k):
				dx[i][j] = (n * z_i[j] - cum_sum) 
		
		else:
			# Use broadcasting
			# without the for loop  for 'j'		
			dx[i] = n*z_i - cum_sum

	return dx
				






