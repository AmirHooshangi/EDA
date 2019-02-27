import tensorflow as tf

class EdaCaeAutoEncoder:
    def __init__(self, number_of_modules, input_size, ae_first_layer_hidden_layer_size
                 , keepprob
                 , input_x):
        self.number_of_modules = number_of_modules
        self.input_size = input_size
        self.ae_first_layer_hidden_layer_size = ae_first_layer_hidden_layer_size
        self.input_x = input_x
        self.keepprob = keepprob
        self.list_of_modules = self.create_modules()

    n1 = 1

    def cae(self, _W, _b, _keepprob):
        _input_r = tf.reshape(self.input_x, shape=[-1, 28, 28, 1])
        # Encoder
        _ce1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d(_input_r, _W['ce1']
                                                 , strides=[1, 2, 2, 1], padding='SAME'), _b['be1']))
        _ce1 = tf.nn.dropout(_ce1, _keepprob)
        pooling = tf.layers.max_pooling2d(_ce1, pool_size=[2, 2], strides=2)
        pooling = tf.nn.lrn(pooling, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        print("pooling ina boodan")
        print(pooling.shape)
        upsample = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(pooling, _W['upe1']
                                                               , tf.stack([tf.shape(self.input_x)[0], 14, 14, 1]),
                                                               strides=[1, 2, 2, 1]
                                                               , padding='SAME'), _b['upb1']))
        print("upsample inas")
        print(upsample.shape)
        _cd1 = tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(upsample, _W['cd1']
                                                           , tf.stack([tf.shape(self.input_x)[0], 28, 28, 1]),
                                                           strides=[1, 2, 2, 1]
                                                           , padding='SAME'), _b['bd1']))
        #   _cd1 = tf.layers.batch_normalization(_cd1)
        _cd1 = tf.nn.dropout(_cd1, _keepprob)
        _out = _cd1
        return {'out': _out}

    def create_modules(self):
        modules = []
        for x in range(0, self.number_of_modules):

            ksize = 9
            weights = {
                'ce1': tf.Variable(tf.random_normal([ksize, ksize, 1, self.n1], stddev=0.1)),
                'cd1': tf.Variable(tf.random_normal([ksize, ksize, 1, self.n1], stddev=0.1))
            }
            biases = {
                'be1': tf.Variable(tf.random_normal([self.n1], stddev=0.1)),

                'bd1': tf.Variable(tf.random_normal([1], stddev=0.1))
            }
            cae = self.cae(weights, biases, self.keepprob)
            modules.append({'cae': cae['out'], 'weights': weights, 'biases': biases, 'encoder': cae['encoder']})
            #mapping_function = (cae)
            #self.list_of_mappings.append(cae)
        return modules

    #    def loss_function(self):
#        average_error = 0
#        for i in range(0, self.number_of_modules):
#            y_prediction = self.list_of_modules[i]
#            average_error += tf.reduce_mean(tf.pow(self.input_x - y_prediction, 2) / self.number_of_modules)
#        return average_error

    
    def loss_function(self):
        a = 0
        b = 0
        c = 0
        average_prediction = 0
        for i in range(0, self.number_of_modules):
            a += tf.pow(self.list_of_modules[i]['cae'] - tf.reshape(self.input_x, shape=[-1, 28, 28, 1]), 2)
            for j in range(0, self.number_of_modules):
                average_prediction += self.list_of_modules[j]['cae']/self.number_of_modules
            b += 0.5 * tf.pow(self.list_of_modules[i]['cae'] - average_prediction, 2)/self.number_of_modules
            c += tf.abs(a - b)/self.number_of_modules
        return tf.reduce_mean(c)


    def get_list_of_modules(self):
        return self.list_of_modules
