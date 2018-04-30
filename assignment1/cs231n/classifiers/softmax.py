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
  N = X.shape[0]
  C = W.shape[1]

  for i in range(N):
      f = X[i].dot(W)
      f -= np.max(f) # to prevent large exp
      denom = np.sum(np.exp(f))
      loss += -f[y[i]] + np.log(denom)

      for j in range(C):
          prob = np.exp(f[j]) / denom
          dW[:, j] += (prob - (j == y[i])) * X[i]

  # average
  loss /= N
  dW /= N

  # regularization
  loss += reg * np.sum(W*W)
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  N = X.shape[0]

  f = X.dot(W)
  f -= np.max(f)
  ENs = np.exp(f)
  denoms = np.sum(ENs, axis=1)
  f_y = f[np.arange(N), y]

  loss = np.sum(-f_y) + np.sum(np.log(denoms))

  probs = ENs / denoms[:, np.newaxis]
  indicator = np.zeros(probs.shape)
  indicator[np.arange(N), y] = 1
  dW = X.T.dot(probs - indicator)

  # average
  loss /= N
  dW /= N

  # regularization
  loss += reg * np.sum(W*W)
  dW += 2 * reg * W

  return loss, dW
