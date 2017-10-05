import cv2
import numpy as np

def SfM (tracked_points):
	# Create the U matrix and V
	U, V = None, None
	for lines in tracked_points:
		U_line = lines[:,0].reshape(1, -1)
		V_line = lines[:,1].reshape(1, -1)
		if U is None: U = U_line
		else: U = np.append(U, U_line, axis=0)
		if V is None: V = V_line
		else: V = np.append(V, V_line, axis=0)
	a = np.empty((U.shape[0], 1))
	for l in range(U.shape[0]):
		a[l] = np.sum(U[l])
	b = np.empty((V.shape[0], 1))
	for l in range(V.shape[0]):
		b[l] = np.sum(V[l])
	U_=U-a
	V_=V-b
	# then append U and V to create W
	#W is the registered measurenment matrix
	W = np.append(U_, V_, axis=0)

	if W.shape[0] < W.shape[1]:
		W = W.transpose()
	
	# Find the factorization of the matrix W
	# Where W = u.s.v
	u, s, v = np.linalg.svd(W, full_matrices=False)
	# get sub matriz of u,s,v
	u3 = u[:,0:3]
	s3 = s[0:3]
	v3 = v[0:3,:]
	# Transform the vector s3 into a 3x3 diagonal matrix
	s3_ = np.power(s3, 1/2)
	sqr_s3_ = np.zeros((3,3))
	for i in range(3): sqr_s3_[i,i] = s3_[i]
	# Find the R' and S' matrix that W'=R'.S'
	R = np.dot(u3, sqr_s3_)
	S = np.dot(sqr_s3_, v3)
	print(R.shape, S.shape)

	# Find L in :
	# ALAt = Id  			# (N, 3) (3, 3) (3, N) = (N, N) 
	# ALAtA = IdA  			# (N, 3) (3, 3) (3, N) (N, 3) = (N, N) (N, 3) ----> (N, 3) (3, 3) (3, 3) = (N, 3)
	# AL = IdA(AtA)⁻1		# (N, 3) (3, 3) = (N, 3) (3, 3) ----> (N, 3) (3, 3) = (N, 3) 
	Rt = R.transpose()
	a_sqr = np.dot(Rt, R)
	Id = np.identity(R.shape[0])
	Id = np.dot(Id, R)
	Id = np.dot(Id, np.linalg.inv(a_sqr))
	L, res, rank, s = np.linalg.lstsq(R, Id)
	print(L.shape)

	#Find Q in: L=QQt
	Q = np.linalg.cholesky(L)
	print(Q.shape)
	
	#Find the Rotation and Shape Matrix
	R_final = np.dot(R, Q)
	S_final = np.dot(np.linalg.inv(Q), S)
	print(R_final.shape, S_final.shape)

	






