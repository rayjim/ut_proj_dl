from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
import numpy as np
import theano
import theano.tensor as T



def homework(train_X, train_y, test_X):
    import theano
    import theano.tensor as T
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    from collections import OrderedDict

    def relu(x):
        return T.maximum(0, x)


    def sigmoid(x):
        return T.nnet.sigmoid(x)


    def tanh(x):
        return T.tanh(x)


    class Metric(object):
    
        def __init__(self, x, y):
            self.x = x
            self.y = y
    
        def negative_log_likelihood(self):
            self.prob_of_y_given_x = T.nnet.softmax(self.x)
            return -T.mean(T.log(self.prob_of_y_given_x)[T.arange(self.y.shape[0]), self.y])
    
        def cross_entropy(self):
            self.prob_of_y_given_x = T.nnet.softmax(self.x)
            return T.mean(T.nnet.categorical_crossentropy(self.prob_of_y_given_x, self.y))
    
        def mean_squared_error(self):
            return T.mean((self.x - self.y) ** 2)
    
        def errors(self):
            if self.y.ndim != self.y_pred.ndim:
                raise TypeError('y should have the same shape as self.y_pred',
                                ('y', self.y.type, 'y_pred', self.y_pred.type))
    
            if self.y.dtype.startswith('int'):
                self.prob_of_y_given_x = T.nnet.softmax(self.x)
                self.y_pred = T.argmax(self.prob_of_y_given_x, axis=1)
                return T.mean(T.neq(self.y_pred, self.y))
            else:
                return NotImplementedError()
    
        def accuracy(self):
            if self.y.dtype.startswith('int'):
                self.prob_of_y_given_x = T.nnet.softmax(self.x)
                self.y_pred = T.argmax(self.prob_of_y_given_x, axis=1)
                return T.mean(T.eq(self.y_pred, self.y))
            else:
                return NotImplementedError()
    
    
    def shared_data(x, y):
        shared_x = theano.shared(
            np.asarray(x, dtype=theano.config.floatX), borrow=True)
        if y is None:
            return shared_x
    
        shared_y = theano.shared(
            np.asarray(y, dtype=theano.config.floatX), borrow=True)
    
        return shared_x, T.cast(shared_y, 'int32')
    
    
    def build_shared_zeros(shape, name):
        """ Builds a theano shared variable filled with a zeros numpy array """
        return theano.shared(
            value=np.zeros(shape, dtype=theano.config.floatX),
            name=name,
            borrow=True
        )
    
    
    def dropout(rng, x, train, p=0.5):
        masked_x = None
        if p > 0.0 and p < 1.0:
            seed = rng.randint(2 ** 30)
            srng = T.shared_randomstreams.RandomStreams(seed)
            mask = srng.binomial(
                n=1,
                p=1.0 - p,
                size=x.shape,
                dtype=theano.config.floatX
            )
            masked_x = x * mask
        else:
            masked_x = x
        return T.switch(T.neq(train, 0), masked_x, x * (1.0 - p))
    
    
    class FullyConnectedLayer(object):
    
        def __init__(self, rng, input=None, n_input=784, n_output=10, activation=None, W=None, b=None):
    
            self.input = input
    
            if W is None:
                W_values = np.asarray(
                    rng.uniform(low=-np.sqrt(6.0 / (n_input + n_output)),
                                high=np.sqrt(6.0 / (n_input + n_output)),
                                size=(n_input, n_output)),
                    dtype=theano.config.floatX)
                if activation == sigmoid:
                    W_values *= 4.0
                W = theano.shared(value=W_values, name='W', borrow=True)
    
            if b is None:
                b_values = np.zeros((n_output,), dtype=theano.config.floatX)
                b = theano.shared(value=b_values, name='b', borrow=True)
    
            self.W = W
            self.b = b
    
            linear_output = T.dot(input, self.W) + self.b
    
            if activation is None:
                self.output = linear_output
            else:
                self.output = activation(linear_output)
    
            self.params = [self.W, self.b]
    
    
    class Optimizer(object):
    
        def __init__(self, params=None):
            if params is None:
                return NotImplementedError()
            self.params = params
    
        def updates(self, loss=None):
            if loss is None:
                return NotImplementedError()
    
            self.updates = OrderedDict()
            self.gparams = [T.grad(loss, param) for param in self.params]
    
    
    def build_shared_zeros(shape, name):
        """ Builds a theano shared variable filled with a zeros numpy array """
        return theano.shared(
            value=np.zeros(shape, dtype=theano.config.floatX),
            name=name,
            borrow=True
        )

    
    class RMSprop(Optimizer):
    
        def __init__(self, learning_rate=0.001, alpha=0.99, eps=1e-8, params=None):
            super(RMSprop, self).__init__(params=params)
    
            self.learning_rate = learning_rate
            self.alpha = alpha
            self.eps = eps
    
            self.mss = [
                build_shared_zeros(t.shape.eval(), 'ms') for t in self.params]
    
        def updates(self, loss=None):
            super(RMSprop, self).updates(loss=loss)
    
            for ms, param, gparam in zip(self.mss, self.params, self.gparams):
                _ms = ms * self.alpha
                _ms += (1 - self.alpha) * gparam * gparam
                self.updates[ms] = _ms
                self.updates[param] = param - self.learning_rate * \
                    gparam / T.sqrt(_ms + self.eps)
    
            return self.updates

    class AdaDelta(Optimizer):
    
        def __init__(self, rho=0.95, eps=1e-6, params=None):
            super(AdaDelta, self).__init__(params=params)
    
            self.rho = rho
            self.eps = eps
            self.accugrads = [
                build_shared_zeros(t.shape.eval(), 'accugrad') for t in self.params]
            self.accudeltas = [
                build_shared_zeros(t.shape.eval(), 'accudelta') for t in self.params]
    
        def updates(self, loss=None):
            super(AdaDelta, self).updates(loss=loss)
    
            for accugrad, accudelta, param, gparam\
                    in zip(self.accugrads, self.accudeltas, self.params, self.gparams):
                agrad = self.rho * accugrad + (1 - self.rho) * gparam * gparam
                dx = - T.sqrt((accudelta + self.eps) / (agrad + self.eps)) * gparam
                self.updates[accudelta] = (
                    self.rho * accudelta + (1 - self.rho) * dx * dx)
                self.updates[param] = param + dx
                self.updates[accugrad] = agrad
    
            return self.updates

    class MomentumSGD(Optimizer):
    
        def __init__(self, learning_rate=0.01, momentum=0.9, params=None):
            super(MomentumSGD, self).__init__(params=params)
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.vs = [build_shared_zeros(t.shape.eval(), 'v')
                       for t in self.params]
    
        def updates(self, loss=None):
            super(MomentumSGD, self).updates(loss=loss)
    
            for v, param, gparam in zip(self.vs, self.params, self.gparams):
                _v = v * self.momentum
                _v = _v - self.learning_rate * gparam
                self.updates[param] = param + _v
                self.updates[v] = _v
    
            return self.updates    
    
    class Adam(Optimizer):
    
        def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, gamma=1 - 1e-8, params=None):
            super(Adam, self).__init__(params=params)
    
            self.alpha = alpha
            self.b1 = beta1
            self.b2 = beta2
            self.gamma = gamma
            self.t = theano.shared(np.float32(1))
            self.eps = eps
    
            self.ms = [build_shared_zeros(t.shape.eval(), 'm')
                       for t in self.params]
            self.vs = [build_shared_zeros(t.shape.eval(), 'v')
                       for t in self.params]
    
        def updates(self, loss=None):
            super(Adam, self).updates(loss=loss)
            self.b1_t = self.b1 * self.gamma ** (self.t - 1)
    
            for m, v, param, gparam \
                    in zip(self.ms, self.vs, self.params, self.gparams):
                _m = self.b1_t * m + (1 - self.b1_t) * gparam
                _v = self.b2 * v + (1 - self.b2) * gparam ** 2
    
                m_hat = _m / (1 - self.b1 ** self.t)
                v_hat = _v / (1 - self.b2 ** self.t)
    
                self.updates[param] = param - self.alpha * \
                    m_hat / (T.sqrt(v_hat) + self.eps)
                self.updates[m] = _m
                self.updates[v] = _v
            self.updates[self.t] = self.t + 1.0
    
            return self.updates

    class MLP(object):
    
        def __init__(self, rng, n_input=784, n_hidden=[500, 500, 500], n_output=10, optimizer=AdaDelta):
    
            self.rng = rng
            self.batchsize = 100
    
            self.n_input = n_input
            self.n_hidden = n_hidden
            self.n_output = n_output
            self.n_layer = len(n_hidden)
    
            self.L1_reg = 0.0
            self.L2_reg = 0.001
    
            """symbol definition"""
            self.index = T.lscalar()
            self.x = T.matrix('x')
            self.y = T.ivector('y')
            self.train = T.iscalar('train')
    
            """network structure definition"""
            self.layers = []
            self.params = []
            for i in range(self.n_layer + 1):
                """for first hidden layer"""
                if i == 0:
                    layer_n_input = self.n_input
                    layer_n_output = self.n_hidden[0]
                    layer_input = dropout(self.rng, self.x, self.train, p=0.1)
                    activation = sigmoid
                elif i != self.n_layer:
                    layer_n_input = self.n_hidden[i - 1]
                    layer_n_output = self.n_hidden[i]
                    layer_input = dropout(
                        self.rng, self.layers[-1].output, self.train)
                    activation = sigmoid
                else:
                    """for output layer"""
                    layer_n_input = self.n_hidden[-1]
                    layer_n_output = self.n_output
                    layer_input = self.layers[-1].output
                    activation = None
    
                layer = FullyConnectedLayer(
                    self.rng,
                    input=layer_input,
                    n_input=layer_n_input,
                    n_output=layer_n_output,
                    activation=activation
                )
                self.layers.append(layer)
                self.params.extend(layer.params)
    
            """regularization"""
            # self.L1 = abs(self.h1.W).sum() + abs(self.pred_y.W).sum()
            # self.L2 = abs(self.h1.W**2).sum() + abs(self.pred_y.W**2).sum()
    
            """loss accuracy error"""
            self.metric = Metric(self.layers[-1].output, self.y)
            # + L1_reg*self.L1 + L2_reg*self.L2
            self.loss = self.metric.negative_log_likelihood()
            self.accuracy = self.metric.accuracy()
            self.errors = self.metric.errors()
    
            """parameters (i.e., weights and biases) for whole networks"""
            # self.params
    
            """optimizer for learning parameters"""
            self.optimizer = optimizer(params=self.params)
    
            """definition for optimizing update"""
            self.updates = self.optimizer.updates(self.loss)
    
        def fine_tuning(self, train_X, train_y, test_X, batchsize=128, n_epoch=100):
    
            self.batchsize = batchsize
            self.n_epoch = n_epoch
    
            """data pre-processing"""
            self.x_train, self.y_train = shared_data(train_X, train_y)
    
            self.x_test = shared_data(test_X, None)
            self.n_train_batches = self.x_train.get_value(
                borrow=True).shape[0] / self.batchsize
    
            # print "# of train mini-batches: " + str(self.n_train_batches)
    
            self.train_model = theano.function(
                inputs=[self.index],
                outputs=[self.loss, self.accuracy],
                updates=self.updates,
                givens={
                    self.x: self.x_train[self.index * self.batchsize: (self.index + 1) * self.batchsize],
                    self.y: self.y_train[self.index * self.batchsize: (self.index + 1) * self.batchsize],
                    self.train: np.cast['int32'](1)
                },
                mode='FAST_RUN'
            )

            self.generate_prediction = theano.function(
                inputs=[],
                outputs=[self.layers[-1].output],
                givens={
                    self.x: self.x_test,
                    self.train: np.cast['int32'](0)
                },
                mode='FAST_RUN'
            )

            epoch = 0
            acc, loss = [], []
            val_acc, val_loss = [], []
            while epoch < self.n_epoch:
                epoch += 1
                acc.append(0.0)
                loss.append(0.0)
                for batch_index in range(self.n_train_batches):
                    batch_loss, batch_accuracy = self.train_model(batch_index)
                    acc[-1] += batch_accuracy
                    loss[-1] += batch_loss
                acc[-1] /= self.n_train_batches
                loss[-1] /= self.n_train_batches
                print ('epoch: {0}, train mean loss={1}, train accuracy={2}'.format(epoch, loss[-1], acc[-1]))

            pred_y_given_x = self.generate_prediction()
            return np.argmax(pred_y_given_x[0], axis=1)

    random_state = 1234
    n_input = train_X.shape[1]
    n_hidden = [500, 500]
    n_output = 10
    rng = np.random.RandomState(random_state)
    mlp = MLP(rng, n_input=n_input, n_hidden=n_hidden,
              n_output=n_output, optimizer=AdaDelta)
    pred_y = mlp.fine_tuning(
        train_X, train_y, test_X, batchsize=100, n_epoch=100)
    return pred_y

if __name__ == '__main__':
    rng = np.random.RandomState(1234)

    # Multi Layer Perceptron
    mnist = fetch_mldata('MNIST original', data_home="..")
    mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                               mnist.target.astype('int32'), random_state=42)

    mnist_X = mnist_X / 255.0

    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y,
                                                        test_size=0.2,
                                                        random_state=412)

    # valid_y = np.eye(10)[valid_y]
    import time
    start = time.time()
    pred_y = homework(train_X, train_y, test_X)
    print ("time is {0}".format(time.time()-start))
    print f1_score(test_y, pred_y, average='macro')