from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        h, h_cache = affine_relu_forward(X, W1, b1)
        scores, f_cache = affine_forward(h, W2, b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        grads = {}
        loss, dL = softmax_loss(scores, y)

        dh, grads['W2'], grads['b2'] = affine_backward(dL, f_cache)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dh, h_cache)

        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        if self.num_layers > 1:
            self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
            self.params['b1'] = np.zeros(hidden_dims[0])
            self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(
                    hidden_dims[-1], num_classes)
            self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
        else:
            self.params['W1'] = weight_scale * np.random.randn(input_dim, num_classes)
            self.params['b1'] = np.zeros(num_classes)

        for i in range(2, self.num_layers):
            self.params['W' + str(i)] = weight_scale * np.random.randn(
                    hidden_dims[i-2], hidden_dims[i-1])
            self.params['b' + str(i)] = np.zeros(hidden_dims[i-1])

        if self.use_batchnorm:
            for i in range(1, self.num_layers):
                self.params['gamma' + str(i)] = np.ones(hidden_dims[i-1])
                self.params['beta' + str(i)] = np.zeros(hidden_dims[i-1])

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        # FORWARD
        cache = {}
        out = X

        for i in range(1, self.num_layers + 1):
            W, b = self.params['W' + str(i)], self.params['b' + str(i)]

            if i < self.num_layers:
                if self.use_batchnorm:
                    gamma = self.params['gamma' + str(i)]
                    beta = self.params['beta' + str(i)]
                    out, cache[i] = affine_batchnorm_relu_forward(out, W, b,
                            gamma, beta, self.bn_params[i-1])
                else:
                    out, cache[i] = affine_relu_forward(out, W, b)
                if self.use_dropout:
                    out, cache[i + 0.5] = dropout_forward(out, self.dropout_param)
            else:
                scores, cache[i] = affine_forward(out, W, b)

        # If test mode return early
        if mode == 'test':
            return scores

        # BACKWARD
        loss, dout = softmax_loss(scores, y)
        grads = {}

        for i in range(self.num_layers, 0, -1):
            if i < self.num_layers:
                if self.use_dropout:
                    dout = dropout_backward(dout, cache[i + 0.5])
                if self.use_batchnorm:
                    dout, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(
                            dout, cache[i])
                    grads['gamma' + str(i)], grads['beta' + str(i)] = dgamma, dbeta
                else:
                    dout, dW, db = affine_relu_backward(dout, cache[i])
            else:
                dout, dW, db = affine_backward(dout, cache[i])

            W = self.params['W' + str(i)]
            loss += 0.5 * self.reg * np.sum(W*W)
            grads['W' + str(i)], grads['b' + str(i)] = dW + self.reg * W, db

        return loss, grads
