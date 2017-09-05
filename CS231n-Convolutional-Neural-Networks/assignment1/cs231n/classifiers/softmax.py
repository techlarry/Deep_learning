import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = np.dot(X, W) # N, C
  for i in xrange(num_train):
    f = scores[i] - np.max(scores[i]) # use a normalization trick
    loss += -np.log(np.exp(f[y[i]])/np.sum(np.exp(f))) # cross-entropy loss
    for j in xrange(num_class):
        softmax_output = np.exp(f[j])/sum(np.exp(f))
        if j == y[i]:
             dW[:,j] += (-1 + softmax_output) *X[i] 
        else: 
             dW[:,j] += softmax_output *X[i] 
  
  loss /= num_train
  dW /= num_train
  loss += reg*np.sum(W*W)
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X, W) # N, C
  f = scores - np.amax(scores, axis=1, keepdims=True) # use a normalization trick
  softmax = -np.log(np.exp(f[range(num_train), y]).reshape(-1,1) / np.sum(np.exp(f), axis=1, keepdims=True))
  loss = np.sum(softmax)/num_train + reg*np.sum(W*W)

  dS = np.exp(f)/np.sum(np.exp(f), axis=1, keepdims=True)
  dS[range(num_train), list(y)] += -1
  dW = np.dot(X.T, dS)/num_train + 2* reg* W 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

