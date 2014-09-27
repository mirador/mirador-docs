# This code implements a neural network predictor. The dependent variable
# must be binary and the first column in the data frame, and all the independent
# categorical variables must be binary as well.
# Requires:
# pandas http://pandas.pydata.org/
# numpy http://www.numpy.org/

import sys
import math
import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt

def thetaMatrix(theta, N, L, S, K):
    # The cost argument is a 1D-array that needs to be reshaped into the
    # parameter matrix for each layer:
    thetam = [None] * L
    C = (S - 1) * N
    thetam[0] = theta[0 : C].reshape((S - 1, N))
    for l in range(1, L - 1):
        thetam[l] = theta[C : C + (S - 1) * S].reshape((S - 1, S))
        C = C + (S - 1) * S
    thetam[L - 1] = theta[C : C + K * S].reshape((K, S))
    return thetam

def gradientArray(gmatrix, N, L, S, K):
    garray = np.zeros((S - 1) * N + (L - 2) * (S - 1) * S + K * S)
    C0 = (S - 1) * N
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.copyto.html
    np.copyto(garray[0 : C0], gmatrix[0].reshape(C0))
    C = C0
    for l in range(1, L - 1):
        Ch = (S - 1) * S
        np.copyto(garray[C : C + Ch], gmatrix[l].reshape(Ch))
        C = C + Ch
    Ck =  K * S
    np.copyto(garray[C : C + Ck], gmatrix[L - 1].reshape(Ck))
    
    return garray

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def forwardProp(x, thetam, L):
    a = [None] * (L + 1)
    a[0] = x
    for l in range(0, L):            
        z = np.dot(thetam[l], a[l])
        res = sigmoid(z)
        a[l + 1] = np.insert(res, 0, 1) if l < L - 1 else res
    return a

def backwardProp(y, a, thetam, L, N):
    err = [None] * (L + 1)
    err[L] = a[L] - y
    for l in range(L - 1, 0, -1):  
        backp = np.dot(np.transpose(thetam[l]), err[l + 1])
        deriv = np.multiply(a[l], 1 - a[l])
        err[l] = np.delete(np.multiply(backp, deriv), 0)
    err[0] = np.zeros(N);
    return err

def cost(theta, X, y, N, L, S, K, gamma):
    M = X.shape[0]

    # The cost argument is a 1D-array that needs to be reshaped into the
    # parameter matrix for each layer:
    thetam = thetaMatrix(theta, N, L, S, K)

    h = np.zeros(M)
    terms = np.zeros(M)
    for i in range(0, M):
        a = forwardProp(X[i,:], thetam, L)
        h[i] = a[L]
        t0 = -y[i] * np.log(h[i]) if 0 < y[i] else 0
        t1 = -(1-y[i]) * np.log(1-h[i]) if y[i] < 1 else 0    
        if math.isnan(t0) or math.isnan(t1):
            #print "NaN detected when calculating cost contribution of observation",i
            terms[i] = 10
        else:
            terms[i] = t0 + t1 
    
    # Regularization penalty
    penalty = (gamma/2) * np.sum(theta * theta)

    return terms.mean() + penalty;

def gradient(theta, X, y, N, L, S, K, gamma):
    M = X.shape[0]

    # The cost argument is a 1D-array that needs to be reshaped into the
    # parameter matrix for each layer:
    thetam = thetaMatrix(theta, N, L, S, K)

    # Init auxiliary data structures
    delta = [None] * L
    D = [None] * L
    delta[0] = np.zeros((S - 1, N))
    D[0] = np.zeros((S - 1, N))
    for l in range(1, L - 1):
        delta[l] = np.zeros((S - 1, S))
        D[l] = np.zeros((S - 1, S))
    delta[L - 1] = np.zeros((K, S))
    D[L - 1] = np.zeros((K, S))

    for i in range(0, M):        
        a = forwardProp(X[i,:], thetam, L)
        err = backwardProp(y[i], a, thetam, L, N)        
        for l in range(0, L):
            # Notes about multiplying numpy arrays: err[l+1] is a 1-dimensional
            # array so it needs to be made into a 2D array by putting inside [],
            # and then transposed so it becomes a column vector.
            prod = np.array([err[l+1]]).T * np.array([a[l]])
            delta[l] = delta[l] + prod

    for l in range(0, L):
        D[l] = (1.0 / M) * delta[l] + gamma * thetam[l]
    
    grad = gradientArray(D, N, L, S, K)

    global gcheck
    if gcheck:
        ok = True
        size = theta.shape[0] 
        epsilon = 1E-5
        maxerr = 0.01
        grad1 = np.zeros(size);
        for i in range(0, size):
            theta0 = np.copy(theta)
            theta1 = np.copy(theta)

            theta0[i] = theta0[i] - epsilon
            theta1[i] = theta1[i] + epsilon

            c0 = cost(theta0, X, y, N, L, S, K, gamma)
            c1 = cost(theta1, X, y, N, L, S, K, gamma)
            grad1[i] = (c1 - c0) / (2 * epsilon)
            diff = abs(grad1[i] - grad[i])
            if maxerr < diff: 
                print "Numerical and analytical gradients differ by",diff,"at argument",i,"/",size
                ok = False
        if ok:
            print "Numerical and analytical gradients coincide within the given precision of",maxerr

    return grad

def predict(x, theta, N, L, S, K):
    thetam = thetaMatrix(theta, N, L, S, K)
    a = forwardProp(x, thetam, L) 
    h = a[L]
    return h;

def debug(theta):
    global params 
    global values
    (X, y, N, L, S, K, gamma) = params
    value = cost(theta, X, y, N, L, S, K, gamma);
    values = np.append(values, [value])
    #print value

# Main -----------------------------------------------------------------------------

# Number of layers (including output) and factor to calculate 
# number of hidden units given the number of input units (variables)
L = 1
hf = 1
K = 1
# Regularization coefficient
gamma = 0
# Default convergence threshold
threshold = 1E-5
showp = False
gcheck = False
testMode = False
testCount = 1

for i in range(1, len(sys.argv)):
    if i == 1:
        filename = sys.argv[1]
    elif  sys.argv[i] == '-layers':
        L = int(sys.argv[i+1])
    elif sys.argv[i] == '-hfactor':
        hf = float(sys.argv[i+1])
    elif sys.argv[i] == '-reg':
        gamma = float(sys.argv[i+1])
    elif sys.argv[i] == '-conv':
        threshold = float(sys.argv[i+1])
    elif sys.argv[i] == '-plot':
        showp = True
    elif sys.argv[i] == '-check':
        gcheck = True
    elif sys.argv[i] == '-test':
        testMode = True
        testCount = int(sys.argv[i+1])

if L < 1:
    print "Need to have at least one hidden layer"
    sys.exit(1)

L = L + 1

# Loading data frame and initalizing dimensions
df = pd.read_csv(filename, delimiter=',', na_values="\\N")
M = df.shape[0]
N = df.shape[1]
S = int((N - 1) * hf) # add 1 for the bias unit on each layer
print 'Number of data samples          :', M
print 'Number of independent variables :', N-1 
print 'Number of hidden layers         :', L-1
print 'Number of units per hidden layer:', S-1
print 'Number of output classes        :', K

y = df.values[:,0]

# Building the (normalized) design matrix
X = np.ones((M, N))
for j in range(1, N):
    # Computing i-th column. The pandas dataframe
    # contains all the values as numpy arrays that
    # can be handled individually:
    values = df.values[:, j]
    minv = values.min()
    maxv = values.max()
    X[:, j] = (values - minv) / (maxv - minv)

rates = np.array([])
for iter in range(0, testCount):
    if testMode: print "-------> Iteration test",iter

    # Create training set by randomly choosing 70% of rows from each output
    # category
    i0 = np.where(y == 0)
    i1 = np.where(y == 1)
    ri0 = np.random.choice(i0[0], size=0.7*i0[0].shape[0], replace=False)
    ri1 = np.random.choice(i1[0], size=0.7*i1[0].shape[0], replace=False)
    itrain = np.concatenate((ri1, ri0))
    itrain.sort()

    Xtrain = X[itrain,:]
    ytrain = y[itrain]

    # Number of parameters:
    # * (S - 1) x N for the first weight natrix (N input nodes counting the bias term), into S-1 nodes in
    #   the first hidden layer
    # * (S - 1) x S for all the weight matrices in the hidden layers, which go from S (counting bias term)
    #   into S-1 nodes in the next layer. Since L counts th number of hidden layers plus the output layer,
    #   and the first transition was accounted by the first term, then we only need L-2
    # * K x S, for the last transition into the output layer with K nodes
    theta0 = np.random.rand((S - 1) * N + (L - 2) * (S - 1) * S + K * S)
    params = (Xtrain, ytrain, N, L, S, K, gamma)

    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html
    print "Training Neural Network..."
    values = np.array([])
    thetaOpt = fmin_bfgs(cost, theta0, fprime=gradient, args=params, gtol=threshold, callback=debug)
    print "Done!"

    # Calculating the prediction rate by applying the trained model on the remaining
    # 30% of the data (the test set), and comparing with random selection
    ntot = 0
    nhit = 0
    nran = 0
    for i in range(0, M):
        if not i in itrain:
            ntot = ntot + 1
            p = predict(X[i,:], thetaOpt, N, L, S, K)
            if (y[i] == 1) == (0.5 < p):
                nhit = nhit + 1
            r = np.random.rand()
            if (y[i] == 1) == (0.5 < r):
                nran = nran + 1

    rate = float(nhit) / float(ntot)
    rrate = float(nran) / float(ntot)
    rates = np.append(rates, [rate])

    print ''
    print '---------------------------------------'
    print 'Predictor success rate on test set:', round(100 * rate, 2), '%'
    print 'Random success rate on test set   :', round(100 * rrate, 2), '%'

    if showp and not testMode:
        plt.plot(np.arange(values.shape[0]), values)
        plt.show()

if testMode:
    print ''
    print '***************************************'
    print 'Average success rate:', round(100 * np.average(rates), 2), '%'
    print 'Standard deviation  :', round(100 * np.std(rates, ddof=1), 2), '%'
