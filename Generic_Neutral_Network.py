# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:55:43 2021

@author: jesse

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import time
import pickle
import numpy.matlib

def create_shapes(size_in, n_labels, n_layers, list):
    # this function turns possible the creation of the random theta initial with init_random_THETA(shapes)
    # Create the shapes list for a desired format of neural network
    # size_in is the number of inputs for the NN (401)
    # n_labels is the number of classifications 
    # n_layers IS THE TOTAL NUMBER OF LAYERS (incluing the example and the classification)
    # list is the number of units per intermediate layer. For ex. if it is a 4 layer NN, I should use list [100, 25] (example)
    n_thetas = n_layers - 2
    shapes = []
    for i in range(n_thetas):
        if i==0:
            shapes.append(( list[i],size_in, list[i]*size_in))
        if i==n_thetas-1:
            shapes.append((n_labels, list[i]+1, (list[i]+1)*n_labels))
        else:
            shapes.append((list[i+1], list[i]+1, (list[i]+1)*list[i+1]))
    return shapes
# DEFINE SETTINGS OF THE NEURAL NETWORK
# size_in must be the number of input units and the n_labels is the number of outputs
# if the NN has more than 1 hidden layer, you can use the same number of hidden units in every layer
# The weights are randomly initialized
# IT is possible to verify the values obtained by the gradient by numerically checking if the
# backpropagation gradient is near to the computed grad [ J(Theta+delta)-J(Theta-delta)]/2/delta
# It is possible to use the gradient descent algorithm, but here the CG method is used to minimize
# the J function (cost function)

maxIterationsNN = 100
n_layers = 4 # soma das layers intermediárias + 2 iniciais e finais
shapes = create_shapes(401, 10, n_layers, [100,100])
lmbda = 1.2

def save_pickle(x,name):
    import pickle
    string = name+".pickle"
    a = open(string, "wb")
    pickle.dump(x, a)
    a.close()
def load_pickle(name):
    import pickle
    string = name+".pickle"
    return pickle.load(open(string,"rb"))
def add_onevec(x):
    ones_vec = np.ones(len(x))
    ones_vec = ones_vec[:,None]
    return np.hstack((ones_vec,x))

# LOAD DATA
X = load_pickle("X")
y = load_pickle("y")
y = y.astype(int)
Xmat = add_onevec(X)

def predict_NN(THETA, Xmat):
    # X is the matrix with ones already added (m, n+1)
    # t1 is the first
    m = Xmat.shape[0]
    L = len(THETA)

    a_vec = []; z_vec = []
    
    for layer in range(L):
        if layer==0:
            z_vec.append(np.dot(Xmat, THETA[layer].T))
            a_vec.append(add_onevec(sigmoid(z_vec[layer])))
        elif layer==L-1:
            z_vec.append(np.dot(a_vec[layer-1], THETA[layer].T))
            a_vec.append(sigmoid(z_vec[layer]))
            hV = a_vec[layer]
        else:
            z_vec.append(np.dot(a_vec[layer-1], THETA[layer].T))
            a_vec.append(add_onevec(sigmoid(z_vec[layer])))
    predict = np.argmax(hV, axis=1)
    return predict#np.argmax(a2,1)

def sigmoid(z):
    return 1/(1+np.exp(-z))
def rowvec(x):
    # reshapes the matrix X into a row vector
    length = x.size
    return np.reshape(x, (length,))


def get_shapes(THETA):
    # returns (n_rows, n_cols, sizes) of THETA
    shapes = [(THETA[i].shape[0], THETA[i].shape[1], THETA[i].size) for i in range(len(THETA))]
    return shapes

def Theta_from_nn(nn, shapes):
    # transforms the theta_array(nn) into a list of Theta Matrices
    # returns THETA
    THETA = [[]]*len(shapes)
    posi = 0
    posf = shapes[0][2]
    totalsize = 0
    for layer in range(len(shapes)):
        totalsize = totalsize + shapes[layer][2]

        THETA[layer] = np.reshape(nn[posi:posf], (shapes[layer][0], shapes[layer][1]))
        
        posi += shapes[layer][2]
        if layer<len(shapes)-1:
            posf += shapes[layer+1][2]
    return THETA

def nn_from_Theta(THETA):
    return np.hstack(tuple([rowvec(THETA[i]) for i in range(len(THETA))]))

def sigmoidGrad(z):
    g = np.zeros(z.size)
    return sigmoid(z)*(1-sigmoid(z))

def init_random_THETA(shapes):
    # give the shapes list of the THETA and it will return 
    # a random nn in the size of THETA
    
    THETA = [[]]*len(shapes)
    for i in range(len(shapes)):
        rows = shapes[i][0] # units in the next layer
        cols = shapes[i][1] # units in the layer
        eps_init = np.sqrt(6)/(np.sqrt(rows+cols))
        THETA[i] = np.random.rand(rows, cols)*2*eps_init - eps_init
    return nn_from_Theta(THETA)

def nnCostFunction2(THETA, Xmat, y, lmbda):
    # y must be adjusted for python (starts from 0)
    m = Xmat.shape[0]
    L = len(THETA)
    K = len(np.unique(y)) #number of labels

    a_vec = []; z_vec = []
    
    for layer in range(L):
        if layer==0:
            z_vec.append(np.dot(Xmat, THETA[layer].T))
            a_vec.append(add_onevec(sigmoid(z_vec[layer])))
        elif layer==L-1:
            z_vec.append(np.dot(a_vec[layer-1], THETA[layer].T))
            a_vec.append(sigmoid(z_vec[layer]))
            hV = a_vec[layer]
        else:
            z_vec.append(np.dot(a_vec[layer-1], THETA[layer].T))
            a_vec.append(add_onevec(sigmoid(z_vec[layer])))
    
    J = -1/m * np.sum([np.sum(np.dot(y==i,np.log(hV[:,i]))) + np.dot((1-(y==i)), np.log(1-hV[:,i])) for i in range(K)])\
            + lmbda/2/m * np.sum([np.trace(np.dot(THETA[i][:,1:].T, THETA[i][:,1:])) for i in range(L)])
    
    
    # CALCULATING THE GRADIENT
    ymat = np.zeros((len(y), K))
    for i in range(m):
        ymat[i, y[i]] = 1
    #predict = np.max(hV, axis=1)
    delta = [np.empty((z_vec[i].shape[1],)) for i in range(L)]
    delta[-1] = hV-ymat
    # delta[0] = delta(2);   z[0] = z(2);  THETA[0] = THETA(1)
    # delta(2) = THETA(2).T * delta(3) .* sigGra(z(2))
    # delta[0] = THETA[1].t * delta[1] .* sigGra(z[0])
    for i in np.flip(range(0,L-1)):
        delta[i] = np.dot(delta[i+1], THETA[i+1][:,1:]) * sigmoidGrad(z_vec[i])
    
    # creating DELTA
    DELTA = [np.array([])]*L
    for i in range(L):
        DELTA[i] = np.zeros((THETA[i].shape[0], THETA[i].shape[1]))
    
    for i in range(L):
        if i==0:
            DELTA[i] = 1/m*np.dot(delta[i].T, Xmat)
        else:
            DELTA[i] = 1/m*np.dot(delta[i].T, a_vec[i-1])
        DELTA[i][:,1:] += lmbda/m*THETA[i][:,1:]
    
    """for ex in range(m):
        ex_a1 = X[ex,:]
        
        for i in range(L):
            ex_deltai = delta[i][ex,:]
            if i==0:
                ex_a1 = np.matlib.repmat(ex_a1,len(ex_deltai),1)
                ex_deltai = (np.matlib.repmat(ex_deltai, ex_a1.shape[1], 1)).T
                DELTA[i] += 1/m*ex_a1*ex_deltai
            else:
                ex_ai = a_vec[i-1][ex,:]
                ex_ai = np.matlib.repmat(ex_ai,len(ex_deltai),1)
                ex_deltai = (np.matlib.repmat(ex_deltai, ex_ai.shape[1], 1)).T
                DELTA[i] += 1/m*ex_ai*ex_deltai"""

    
    print(1)
    return J, DELTA


def train_NeuralNetwork(Xmat, y, shapes, lmbda, maxIter):
    
    def foward_propag(THETA,Xmat,L):
        a_vec = []; z_vec = []
        
        # Foward propagation: calculating z, a=sigmoid(z) and h
        for layer in range(L):
            if layer==0:
                z_vec.append(np.dot(Xmat, THETA[layer].T))
                a_vec.append(add_onevec(sigmoid(z_vec[layer])))
            elif layer==L-1:
                z_vec.append(np.dot(a_vec[layer-1], THETA[layer].T))
                a_vec.append(sigmoid(z_vec[layer]))
                hV = a_vec[layer]
            else:
                z_vec.append(np.dot(a_vec[layer-1], THETA[layer].T))
                a_vec.append(add_onevec(sigmoid(z_vec[layer])))
        return a_vec, z_vec, hV
    
    def Cost(nn_theta):
        # REGULARIZED
        # returns the cost function for a regularized neural network
        THETA = Theta_from_nn(nn_theta, shapes)
        m = Xmat.shape[0]
        L = len(THETA)
        K = len(np.unique(y)) #number of labels

        hV = foward_propag(THETA,Xmat,L)[2]
        # Computing the cost
        J = -1/m * np.sum([np.sum(np.dot(y==i,np.log(hV[:,i]))) + np.dot((1-(y==i)), np.log(1-hV[:,i])) for i in range(K)])\
            + lmbda/2/m * np.sum([np.trace(np.dot(THETA[i][:,1:].T, THETA[i][:,1:])) for i in range(L)])
        
        print('Cost computed: {:.4f}.'.format(J))
        # J is a real number. Return it
        return J

    def Cost_prime(nn_theta):
        # REGULARIZED 
        # returns the gradient of the cost function for logistic regression
        
        THETA = Theta_from_nn(nn_theta, shapes)
        m = Xmat.shape[0]
        L = len(THETA)
        K = len(np.unique(y)) #number of labels
        
        a_vec, z_vec, hV = foward_propag(THETA,Xmat,L)
        
        # Foward propagation:
        ymat = np.zeros((len(y), K))
        for i in range(m):
            ymat[i, y[i]] = 1
        #predict = np.max(hV, axis=1)
        delta = [np.empty((z_vec[i].shape[1],)) for i in range(L)]
        delta[-1] = hV-ymat
        
        for i in np.flip(range(0,L-1)):
            delta[i] = np.dot(delta[i+1], THETA[i+1][:,1:]) * sigmoidGrad(z_vec[i])
        
        # creating DELTA (the gradient)
        DELTA = [np.array([])]*L
        for i in range(L):
            DELTA[i] = np.zeros((THETA[i].shape[0], THETA[i].shape[1]))
        
        for i in range(L):
            if i==0:
                DELTA[i] = 1/m*np.dot(delta[i].T, Xmat)
            else:
                DELTA[i] = 1/m*np.dot(delta[i].T, a_vec[i-1])
            DELTA[i][:,1:] += lmbda/m*THETA[i][:,1:]
        
        nn_DELTA = nn_from_Theta(DELTA)
        
        # Return the gradient in a vector form
        return nn_DELTA       
    
    nn_theta = init_random_THETA(shapes) #initializes theta vector (also called nn)
    
    # Call the minimize function to solve the problem:
    sol_by_min = minimize(Cost, nn_theta, method='CG', jac=Cost_prime, options={'maxiter':maxIter,'disp':True})
    return sol_by_min.x

# create_shapes(401, 10, 3, [25])
#shapes = create_shapes(51, 2, 3, [25])

#Solution = train_NeuralNetwork(Xmat[1000:2000,200:251], y[1000:2000]-2, shapes, lmbda)
Solution = train_NeuralNetwork(Xmat, y-1, shapes, lmbda, maxIterationsNN)
THETA_sol = Theta_from_nn(Solution, shapes)
Prediction = predict_NN(THETA_sol, Xmat)
y2 = y-1
accuracy = np.mean(Prediction==y2)
print('---------------------------------------------------------')
print('The accuracy obtained from the '+ str(n_layers) +' layers Neural Network was {:.2f}%.'.format(accuracy*100))
print('---------------------------------------------------------')
for i in range(len(THETA_sol)):
    name = 'Theta'+str(i+2)
    save_pickle(THETA_sol[i], name)
    
"""THETA = [Theta1, Theta2]
nn_params = np.hstack(tuple([rowvec(THETA[i]) for i in range(len(THETA))]))
Theta_shapes = get_shapes(THETA)
Theta_from_nn(nn_params, Theta_shapes)"""





J = nnCostFunction2(THETA, Xmat, y-1, lambdaV)
print('J obtained was: {:.4f}'.format(J))
print('J obtained was: {:.4f}'.format(J))