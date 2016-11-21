'''
Created on Nov 20, 2016

@author: raybao'''


from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

trng = RandomStreams(42)
rng = np.random.RandomState(1234)


# Multi Layer Perceptron
class Layer:
    # Constructor
    def __init__(self, in_dim, out_dim, function):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.function = function
        self.z = T.ivector("z")
        self.W = theano.shared(rng.uniform(low=-0.08, high=0.08,
                                           size=(in_dim, out_dim)
                                           ).astype('float32'), name='W')
        self.b = theano.shared(np.zeros(out_dim).astype('float32'), name='b')
        self.params = [self.W, self.b]

    # Forward Propagation
    def f_prop(self, x):
        self.z = self.function(T.dot(x, self.W) + self.b)
        return self.z


# Stochastic Gradient Descent
def sgd(params, g_params, eps=np.float32(0.1)):
    updates = OrderedDict()
    for param, g_param in zip(params, g_params):
        updates[param] = param - eps*g_param
    return updates


# Momentum
def sgd_momentum(params, g_params, epoch = 0, eps=np.float32(0.1), mom_start=0.9,
                 mom_end=0.5):
    # ... and allocate mmeory for momentum'd versions of the gradient
    learning_rate = eps
    gparams_mom = []
    for param in params:
        gparam_mom = theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                    dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)
    mom_epoch_interval = 500.0
    squared_filter_length_limit = 15.0
    # Compute momentum for the current epoch
    mom = ifelse(epoch < mom_epoch_interval,
                 mom_start*(1.0 - epoch/mom_epoch_interval) + mom_end *
                 (epoch/mom_epoch_interval), np.float32(mom_end))

    # Update the step direction using momentum
    updates = OrderedDict()
    for gparam_mom, gparam in zip(gparams_mom, g_params):

        # change the update rule to match Hinton's dropout paper
        updates[gparam_mom] = mom * gparam_mom - (1. - mom) * \
                                learning_rate * gparam

    # ... and take a step along that direction
    for param, gparam_mom in zip(params, gparams_mom):
        # Misha Denil's original version
        # stepped_param = param - learning_rate * updates[gparam_mom]
        # since we have included learning_rate in gparam_mom, we don't need it
        # here
        stepped_param = param + updates[gparam_mom]

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices.  This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        if param.get_value(borrow=True).ndim == 2:
            # squared_norms = T.sum(stepped_param**2, axis=1).reshape((stepped_param.shape[0],1))
            # scale = T.clip(T.sqrt(squared_filter_length_limit / squared_norms), 0., 1.)
            # updates[param] = stepped_param * scale
            # constrain the norms of the COLUMNs of the weight, according to
            # https://github.com/BVLC/caffe/issues/109
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0,
                                   T.sqrt(squared_filter_length_limit))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')


def test_mlp(datasets):

    train_X, train_y = shared_dataset(datasets[0])
    valid_X, aa = shared_dataset(datasets[1])
    valid_y = datasets[1][1]
    # test_X, test_y = datasets[2]
    layers = [
        Layer(784, 500, T.nnet.sigmoid),
        Layer(500, 500, T.nnet.sigmoid),
        Layer(500, 10, T.nnet.softmax)
    ]

    x = T.fmatrix('x')
    t = T.ivector('t')
    epoch = T.scalar('epoch')
    index = T.lscalar()

    params = []
    for i, layer in enumerate(layers):
        params += layer.params
        if i == 0:
            layer_out = layer.f_prop(x)
        else:
            layer_out = layer.f_prop(layer_out)

    y = layers[-1].z
    # cost = T.mean(T.nnet.categorical_crossentropy(y, t))
    cost = -T.mean(T.log(y)[T.arange(t.shape[0]), t])
    g_params = T.grad(cost=cost, wrt=params)
    updates = sgd_momentum(params, g_params, epoch)

    batch_size = 100

    train = theano.function(inputs=[epoch, index], outputs=[cost],
                            updates=updates,
                            givens={
                            x: train_X[index * batch_size:(index + 1) * batch_size],
                            t: train_y[index * batch_size:(index + 1) * batch_size]},
                            on_unused_input='warn')

    valid = theano.function(inputs=[],
                            outputs=y,
                            givens={
                                x: valid_X},
                             on_unused_input='warn')
    # test = theano.function(inputs=[x], outputs=T.argmax(y, axis=1),\
                            # name='test')

    n_batches = (train_X.get_value(borrow=True).shape[0]//batch_size)
    print n_batches
    for epoch in range(1, 101):
        # train_X, train_y = shuffle(train_X, train_y)
        for index in xrange(n_batches):
            train(epoch, index)
        pred_y = valid()
        print np.argmax(pred_y, axis=1)
        #print valid_y.get_value()
        print('EPOCH:: %i, Validation F1: %.3f' %
              (epoch + 1,
               f1_score(valid_y,
                        np.argmax(pred_y, axis=1), average='macro')))


def load_mnist():
        mnist = fetch_mldata('MNIST original')
        mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                                   mnist.target.astype('int32'),
                                   random_state=42)

        mnist_X = mnist_X / 255.0

        train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y,
                                                            test_size=0.2,
                                                            random_state=42)
        #train_y = np.eye(10)[train_y]
        train_X, test_X, train_y, test_y = train_test_split(train_X, train_y,
                                                            test_size=0.2,
                                                            random_state=42)

        return [(train_X, train_y), (test_X, test_y)]


if __name__ == '__main__':
    datasets = load_mnist()
    test_mlp(datasets)
