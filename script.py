import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


# The mean matrix is a 2 * 5 matrix
# The entry on column 0 are the mean values of X where trueLabel == 1
# e.g. means[0][0] = mean of all entries in X where along column 0, and trueLabel == 1
# means[1][0] == mean of all entries in X where along column 1, and trueLabel == 1
def findMeanOnGivenAxisByClass(X,y,k):
	means = np.zeros([X.shape[1],k], dtype=float)
	for i in range(1,k+1):
		row_num_with_i = np.where(y==i)[0] #all the rows from y where the value == i
		trainMatrix    = X[row_num_with_i,:] # all the rows from X where the true label of the entry == i
		#print "trainMatrix :",trainMatrix
		means[:,i-1]   = np.mean(trainMatrix,axis=0).transpose() #axis=0 means along vertical axis,
		#print "means :",means[:,i-1]
	return means

def findClassesInY(y):
	k = np.unique(y)
	return k,len(k)
#
def ldaLearn(X,y):
	#print X
	# print X.shape
	# print y.shape
	N = X.shape[0] #number of elements
	d = X.shape[1] #number of attributes
	uniqueArray,k = findClassesInY(y); #number of classes
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # IMPLEMENT THIS METHOD
	means = np.zeros([d,k], dtype=float) # 2*5 matrix
	covmat= np.array([d,d], dtype=float) #2*2 matrix

	means = findMeanOnGivenAxisByClass(X,y,k)
	covmat = np.cov(np.transpose(X))
	return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD
	#N =
	means   = np.array([])
	covmats = np.array([])
	return means,covmats

""""
Generates the predicted label from the p(x) formula
Refer eqn 3.1 in slide B.3 for formula
"""
def getPredictedLabel(testX_transpose,means,invCov,numClass,constantTerm):
	prediction = 0
	pdf = 0.0
	res = 0.0
	pdfList = []
	for j in range(numClass): #for all columns in means matrix
		row = testX_transpose - means[:,j]
		res = constantTerm * np.exp((-1/2) * np.dot( np.dot( np.transpose(row) , invCov)  ,(row) ))
		pdfList.append(res)
	prediction = pdfList.index(max(pdfList)) + 1
	return prediction

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
	#X = 100*2 matrix, Y = 100*1 matrix
	#means = 2*5 , covmat = 2*2 matrix
	acc   =0.0;
	N 	  = Xtest.shape[0] # number of test examples
	numClass = means.shape[1] # number of means
	ypred = np.zeros([N,1], dtype = float);
	invCov= inv(covmat) #compute inverse(sigma)

	constantTerm = 1 / (sqrt(np.power(2 * pi,numClass))  * sqrt(det(covmat))) #the constant term in p(x) formula

	"""The idea is to look at all the test examples given in Xtest, and apply the formula for computing p(x)
	#to the dataset
	The predicted label will be the one where the p(x) is maximum (different values of means will be tried out)
	NOTE : We have ignored the term det(covar) * 1/ pow(root(2*pi),d)
	 	   We can do this as this will be the same for all predictions, and hence does not affect our predictions"""

	for i in range(N): #for all training rows
		pdf = 0
		testX_transpose = np.transpose(Xtest[i,:]) #required for (X-mu) operation
		prediction = getPredictedLabel(testX_transpose,means,invCov,numClass,constantTerm)
		#print "prediction ",prediction
		if (prediction == ytest[i]): #check if prediction matches
			acc += 1 #correct prediction
			ypred[i] = prediction
	acc = acc / N #prediction accuracy percentage
	return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
	acc =0
	ypred = np.array([])
	return acc,ypred

def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD
	w = np.array([])
	return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    # Output:
    # w = d x 1

    # IMPLEMENT THIS METHOD
	w = np.array([])
	return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse

    # IMPLEMENT THIS METHOD
	rmse = np.array([])
	return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda

    # IMPLEMENT THIS METHOD

    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    # IMPLEMENT THIS METHOD
    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

plt.show()

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])))
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)

# Problem 2

if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    rmses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
