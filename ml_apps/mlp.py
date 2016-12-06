'''
Created on Nov 19, 2016

@author: raybao
'''


def homework(train_X, train_y, test_X):
    import numpy as np
    from scipy.special import expit

    def softmax(x):
        c = np.max(x)
        exp_x = np.exp(x-c)
        sum_exp_x = np.sum(exp_x, axis=1)
        y = np.divide(exp_x, sum_exp_x[:, None])
        return y

    def deriv_softmax(x):
        return softmax(x)*(1 - softmax(x))

    def sigmoid(x):
        return expit(x)

    def deriv_sigmoid(x):
        return sigmoid(x)*(1 - sigmoid(x))

    def one_hot(x, n_out=10):
        """
        Description:
            get one-hot expression from category expression
        Arguments:
            x: one-dimensional numpy array, data type integer
        Return:
            numpy array, one-hot expression of x
        """
        max_value = n_out - 1  # np.max(x)
        output = np.zeros((x.shape[0], max_value+1))
        output[np.arange(x.shape[0]), x] = 1
        return output

    # Layer
    class Layer:
        # Constructor
        def __init__(self, in_dim, out_dim, function, deriv_function):
            self.W = np.random.uniform(low=-0.08, high=0.08,
                                       size=(in_dim, out_dim)).\
                                       astype("float32")
            self.b = np.zeros(out_dim).astype("float32")
            self.v_W = np.zeros_like(self.W)
            self.v_b = np.zeros_like(self.b)
            self.function = function
            self.deriv_function = deriv_function
            self.u = None
            self.delta = None
            self.in_dim = in_dim
            self.out_dim = out_dim

        def f_prop(self, x):
            self.u = x.dot(self.W) + self.b
            self.z = self.function(self.u)
            return self.z

        # Back Propagation
        def b_prop(self, delta, W, batch_size=1):
            self.delta = self.deriv_function(self.u)*np.dot(delta, W.T)
            return self.delta

    class MLP:

        def __init__(self, in_dim, out_dim, ls_layers=[1000],
                     activation_func="sigmoid", momentum=0.9):

            self.in_dim = in_dim
            self.out_dim = out_dim
            self.activation_function = activation_func
            self.n_layer = len(ls_layers)+1
            self.momentum = momentum
            if activation_func == "sigmoid":
                func = sigmoid
                deriv_func = deriv_sigmoid
            else:
                raise NotImplementedError("we are currently not supported other \
                                            activation function")

            self.layers_dropout = []
            # added sigmoid layer
            for i_layer in range(len(ls_layers)):
                if i_layer == 0:
                    sigmoid_layer = Layer(self.in_dim, ls_layers[i_layer],
                                          func, deriv_func)
                    self.layers_dropout.append(sigmoid_layer)
                else:
                    sigmoid_layer = Layer(self.layers_dropout[i_layer-1].out_dim,
                                          ls_layers[i_layer], func, deriv_func)
                    self.layers_dropout.append(sigmoid_layer)

            # added softmax layer
            softmax_layer = Layer(self.layers_dropout[-1].out_dim, self.out_dim,
                                  softmax, deriv_softmax)
            self.layers_dropout.append(softmax_layer)

        def f_props(self, x):
            z = x
            for layer in self.layers_dropout:
                z = layer.f_prop(z)
            return z

        def b_props(self, delta):
            for i, layer in enumerate(self.layers_dropout[::-1]):
                if i == 0:
                    layer.delta = delta
                else:
                    delta = layer.b_prop(delta, _W)

                _W = layer.W

        def train_one_epoch(self, X, t, eps=0.1, batch_size=1):
            # Forward Propagation
            y = self.f_props(X)
            t = one_hot(t, n_out=self.out_dim)
            # Cost Function & Delta
            # Negative Loglikelihood

            cost = -np.mean(t*np.log(y))

            delta = y - t

            # Back Propagation
            self.b_props(delta)

            # Update Parameters
            z = X
            for i, layer in enumerate(self.layers_dropout):
                dW = z.T.dot(layer.delta)
                db = np.ones(len(z)).dot(layer.delta)
                layer.v_W = self.momentum*layer.v_W - eps*dW/float(batch_size)
                layer.W = layer.W + layer.v_W
                layer.v_b = self.momentum*layer.v_b - eps*db/float(batch_size)
                layer.b = layer.b + layer.v_b
                z = layer.z
            return cost

        def train(self, X, t, eps=0.1, n_epoch=10, batch_size=10):
            # Online Learning
            n_batch = int(X.shape[0]/batch_size)
            for i_iter in range(n_epoch):
                cost = 0
                eps = 1*eps
                for ii in range(n_batch):
                    cost = cost + self.train_one_epoch(X[ii*batch_size:(ii+1) *
                                                         batch_size, :],
                                                       t[ii*batch_size:(ii+1) *
                                                         batch_size],
                                                       batch_size=batch_size,
                                                       eps=eps)
                print ("cost at {} iteration is {}".format(i_iter, cost))

        def predict(self, X):
            self.y_given_x = self.f_props(X)
            y = np.argmax(self.y_given_x, axis=1)
            return y
    n_in = train_X.shape[1]
    ls_layers = [500, 500]
    n_out = 10
    mlp = MLP(n_in, n_out, ls_layers=ls_layers)
    mlp.train(train_X, train_y, n_epoch=50, batch_size=10)
    pred_y = mlp.predict(test_X)

    return pred_y

if __name__ == '__main__':  # pragma: no coverage

    from sklearn.utils import shuffle
    from sklearn.metrics import f1_score
    from sklearn.datasets import fetch_mldata
    from sklearn.cross_validation import train_test_split

    def load_mnist():
        mnist = fetch_mldata('MNIST original')
        mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                                   mnist.target.astype('int32'),
                                   random_state=42)

        mnist_X = mnist_X / 255.0

        train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y,
                                                            test_size=0.2,
                                                            random_state=42)

        return (train_X, test_X, train_y, test_y)

    def check_homework():
        train_X, test_X, train_y, test_y = load_mnist()
        # validate for small dataset
        train_X_mini = train_X
        train_y_mini = train_y
        test_X_mini = test_X
        test_y_mini = test_y

        pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
        return f1_score(test_y_mini, pred_y, average='macro')
    import time
    start = time.time()
    print "f1 score is {}".format(check_homework())
    print "time is {}".format(time.time()-start)
