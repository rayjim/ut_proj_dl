def homework(train_X, train_y, test_X):
    import time
    start_time = time.time()

    train_y = np.eye(10)[train_y].astype('int32')

    train_X = train_X.reshape((train_X.shape[0], 3, 32, 32))
    test_X = test_X.reshape((test_X.shape[0], 3, 32, 32))

    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                          test_size=0.1,
                                                          random_state=42)

    padded = np.pad(train_X, ((0, 0), (0, 0), (4, 4), (4, 4)), mode='constant')
    flip_train_X = train_X[:, :, :, ::-1]
    # flip_train_X_up = train_X[:, :, ::-1, :]
    train_y_single = train_y
    n_sample = len(train_X)
    for ii in range(3):
        crops = np.random.randint(8, size=(n_sample, 2))
        cropped_train_X = [padded[i, :, c[0]:(c[0]+32), c[1]:(c[1]+32)]
                           for i, c in enumerate(crops)]
        cropped_train_X = np.array(cropped_train_X)
        train_X = np.concatenate((train_X, cropped_train_X), axis=0)
        train_y = np.concatenate((train_y, train_y_single), axis=0)

    train_X = np.concatenate((train_X, flip_train_X), axis=0)
    train_y = np.concatenate((train_y, train_y_single), axis=0)
    # train_X = np.concatenate((train_X, flip_train_X_up), axis=0)
    # train_y = np.concatenate((train_y, train_y_single), axis=0)

    def gcn(x):
        mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
        std = np.std(x, axis=(1, 2, 3), keepdims=True)
        return (x - mean)/std
    
    normalized_train_X = gcn(train_X)
    test_X = gcn(test_X)
    valid_X = gcn(valid_X)
    
    class ZCAWhitening:
        def __init__(self, epsilon=1e-4):
            self.epsilon = epsilon
            self.mean = None
            self.ZCA_matrix = None

        def fit(self, x):
            x = x.reshape(x.shape[0], -1)
            self.mean = np.mean(x, axis=0)
            x -= self.mean
            cov_matrix = np.dot(x.T, x) / x.shape[0]
            A, d, _ = np.linalg.svd(cov_matrix)
            self.ZCA_matrix = np.dot(np.dot(A,
                                            np.diag(1. / np.sqrt(d + self.epsilon))
                                            ), A.T)

        def transform(self, x):
            shape = x.shape
            x = x.reshape(x.shape[0], -1)
            x -= self.mean
            x = np.dot(x, self.ZCA_matrix.T)
            return x.reshape(shape)
    
    # 可視化用に各画像ごとに[0,1]にする(ZCAの結果を見やすくするため)&次元を変更
    def deprocess_each_img(x):
        _min = np.min(x, axis=(1, 2, 3), keepdims=True)
        _max = np.max(x, axis=(1, 2, 3), keepdims=True)
        _x = (x - _min)/(_max - _min)
        _x = _x.transpose((0, 2, 3, 1))
        return _x
    # _zca_train_X = deprocess_each_img(zca_train_X)
    
    class BatchNorm:
        # Constructor
        def __init__(self, shape, epsilon=np.float32(1e-5)):
            self.shape = shape
            self.epsilon = epsilon

            self.gamma = theano.shared(np.ones(self.shape, dtype="float32"),
                                       name="gamma")
            self.beta = theano.shared(np.zeros(self.shape, dtype="float32"),
                                      name="beta")
            self.params = [self.gamma, self.beta]

        # Forward Propagation
        def f_prop(self, x):
            if x.ndim == 2:
                mean = T.mean(x, axis=0, keepdims=True)
                std = T.sqrt(T.var(x, axis=0, keepdims=True) + self.epsilon)
            elif x.ndim == 4:
                mean = T.mean(x, axis=(0, 2, 3), keepdims=True)
                std = T.sqrt(T.var(x, axis=(0, 2, 3), keepdims=True) +
                             self.epsilon)

            normalized_x = (x - mean) / std
            self.z = self.gamma * normalized_x + self.beta
            return self.z
    def relu(x):
        return T.maximum(0, x)
    class Conv:
        # Constructor
        def __init__(self, filter_shape, function=lambda x: x, border_mode="valid",
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
    class Activation:
        # Constructor
        def __init__(self, function):
            self.function = function
            self.params = []

        # Forward Propagation
        def f_prop(self, x):
            self.z = self.function(x)
            return self.z
        
    activation = relu
    layers = [                               # (チャネル数)x(縦の次元数)x(横の次元数)
        Conv((32, 3, 3, 3)),                 # 3x32x32  ->  32x30x30
        BatchNorm((32, 30, 30)),
        Activation(activation),
        Pooling((2, 2)),                     # 32x30x30 ->  32x15x15
        Conv((64, 32, 3, 3)),                # 32x15x15 ->  64x13x13
        BatchNorm((64, 13, 13)),
        Pooling((2, 2)),                     # 64x13x13 ->  64x6x6
        Conv((128, 64, 3, 3)),               # 64x6x6   -> 128x4x4
        BatchNorm((128, 4, 4)),
        Activation(activation),
        Pooling((2, 2)),                     # 128x4x4  -> 128x2x2
        Flatten(2),
        Layer(128*2*2, 256, activation),
        Layer(256, 10, T.nnet.softmax)
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
    opt_grad = Adam(params=params)
    updates = opt_grad.updates(cost)

    train = theano.function(inputs=[x, t, train_flag], outputs=cost, updates=updates,
                            allow_input_downcast=True, name='train', mode='FAST_RUN')
    valid = theano.function(inputs=[x, t,train_flag], outputs=[cost, T.argmax(y, axis=1)],
                        allow_input_downcast=True, name='valid', mode='FAST_RUN')
    test = theano.function(inputs=[x,train_flag], outputs=T.argmax(y, axis=1), 
                           name='test', mode='FAST_RUN')
        
    batch_size = 100
    train_X = normalized_train_X
    # print (train_X.shape)
    n_batches = train_X.shape[0]//batch_size
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 1.005  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    best_valid_score = -np.inf
    best_iter = 0

    epoch = 0
    done_looping = False
    
    for epoch in range(20):
        if done_looping is True:
            break
        train_X, train_y = shuffle(train_X, train_y)
        for i in range(n_batches):
            start = i*batch_size
            end = start + batch_size
            cost = train(train_X[start:end], train_y[start:end],1)
        iter = (epoch -1)*n_batches+start
        print('Training  cost: %.3f' % cost)
        valid_cost, pred_y = valid(valid_X, valid_y, 0)
        valid_score = f1_score(np.argmax(valid_y, axis=1).astype('int32'),
                    pred_y, average='macro')
        print('EPOCH:: %i, Validation cost: %.3f, Validation F1: %.3f' %
          (epoch + 1, valid_cost, valid_score))
        if valid_score > best_valid_score:
            if valid_score > best_valid_score*improvement_threshold:
                patience = max(patience, iter*patience_increase)
            best_iter = iter
            best_valid_score = valid_score
            print ('epoch %i, best valid score is %f'%(epoch, best_valid_score))
            
        if patience <=iter:
            done_lopping = True
            break
        # print('EPOCH:: %i, train cost: %f' %(epoch + 1, cost))   
        if time.time()-start_time>3200:
            break
    pred_y = test(test_X,0)
    return pred_y
