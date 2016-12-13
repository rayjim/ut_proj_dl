def homework(train_X, train_y, test_X):
    import theano
    import theano.tensor as T
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    from collections import OrderedDict
    import time
    import numpy as np
    start_time = time.time()
    rng = np.random.RandomState(1234)
    def relu(x):
        return T.maximum(0, x)
    class Conv:
        # Constructor
        def __init__(self, filter_shape, function, border_mode="valid",
                    subsample=(1, 1)):
            self.function = function
            self.border_mode = border_mode
            self.subsample = subsample

            self.W = theano.shared(rng.uniform(low=-0.08, high=0.08,
                                                size=filter_shape
                                                ).astype('float32'), name='W')
            self.b = theano.shared(np.zeros((filter_shape[0],), dtype="float32"),
                                    name="b")  # バイアスはフィルタごと
            self.params = [self.W, self.b]

        # Forward Propagation
        def f_prop(self, x):
            conv_out = conv2d(x, self.W, border_mode=self.border_mode,
                                subsample=self.subsample)  # 畳み込み処理
            # バイアスを加えて（第1要素）活性化関数をかける
            self.z = self.function(conv_out +
                                    self.b[np.newaxis, :, np.newaxis, np.newaxis])
            return self.z   
    
    class Pooling:
        # Constructor
        def __init__(self, pool_size=(2, 2), mode='max'):
            self.pool_size = pool_size
            self.mode = 'max'
            self.params = []

        # Forward Propagation
        def f_prop(self, x):
            return pool.pool_2d(input=x, ds=self.pool_size, mode=self.mode,
                                ignore_border=True)
    class Flatten:
        # Constructor
        def __init__(self, outdim=2):
            self.outdim = outdim
            self.params = []

        # Forward Propagation
        def f_prop(self, x):
            return T.flatten(x, self.outdim)
        

    
    class Layer:
        # Constructor
        def __init__(self, in_dim, out_dim, function):
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.function = function

            self.W = theano.shared(rng.uniform(
                        low=-np.sqrt(6. / (in_dim + out_dim)),
                        high=np.sqrt(6. / (in_dim + out_dim)),
                        size=(in_dim, out_dim)
                    ).astype("float32"), name="W")
            self.b = theano.shared(np.zeros(out_dim).astype("float32"), name="b")
            self.params = [self.W, self.b]

        # Forward Propagation
        def f_prop(self, x):
            self.z = self.function(T.dot(x, self.W) + self.b)
            return self.z


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
        
    activation = relu
    
    layers = [                             # (チャネル数)x(縦の次元数)x(横の次元数)
        Conv((20, 1, 5, 5), activation),   # 1x28x28  -> 20x24x24
        Pooling((2, 2)),                    # 20x24x24 -> 20x12x12
        Conv((50, 20, 5, 5), activation),  # 20x12x12 -> 50x 8x 8
        Pooling((2, 2)),                   # 50x 8x 8 -> 50x 4x 4
        Flatten(2),
        Layer(4*4*50, 500, activation),
        Layer(500, 500, activation),
        Layer(500, 10, T.nnet.softmax)
    ]
    
    def dropout(x, train, p=0.5):
        """
        train: mode
        0 - train
        1 - test
        2 - pretrain
        """
        rng = np.random.RandomState(1234)
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
    

    x = T.ftensor4('x')
    t = T.imatrix('t')
    train_flag = T.iscalar('train_flag')

    params = []
    layer_out = x
    for layer in layers[:-3]:
        params += layer.params
        layer_out = layer.f_prop(layer_out)

    params += layers[-3].params
    input_drop = dropout(layer_out, train_flag, p = 0.1)
    layer_out = layers[-3].f_prop(input_drop)
    params += layers[-2].params
    input_drop = dropout(layer_out, train_flag)
    layer_out = layers[-2].f_prop(input_drop)
    params += layers[-1].params
    layer_out = layers[-1].f_prop(layer_out)
    

    y = layers[-1].z

    cost = T.mean(T.nnet.categorical_crossentropy(y, t))

    # g_params = T.grad(cost, params)
    # updates = sgd(params, g_params)
    opt_grad = AdaDelta(params=params)
    updates = opt_grad.updates(cost)

    train = theano.function(inputs=[x, t, train_flag], outputs=cost, updates=updates,
                            allow_input_downcast=True, name='train', mode='FAST_RUN')
    test = theano.function(inputs=[x,train_flag], outputs=T.argmax(y, axis=1), 
                           name='test', mode='FAST_RUN')
        
    batch_size = 100
    n_batches = train_X.shape[0]//batch_size
    train_y = np.eye(10)[train_y]
    train_X = train_X.reshape((train_X.shape[0], 1, 28, 28))
    test_X = test_X.reshape((test_X.shape[0], 1, 28, 28))
    for epoch in range(1000):
        train_X, train_y = shuffle(train_X, train_y)
        for i in range(n_batches):
            start = i*batch_size
            end = start + batch_size
            cost = train(train_X[start:end], train_y[start:end],1)
        # print('EPOCH:: %i, train cost: %.3f' %
          #    (epoch + 1, cost))   
        if time.time()-start_time>3200:
            break
    pred_y = test(test_X,0)
    return pred_y

