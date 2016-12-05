from __future__ import print_function

__docformat__ = 'restructedtext en'

import theano
import pickle
import os
import sys
import timeit
import random

import theano.tensor as T
import numpy as np

def num_to_list(num, dim):
    '''
    Here num denote a configuration of the 2D flip game in the sense that 
    integer i corresponds to the configuration bin(i)[2:].zfill(dim)[::-1].reshape((row, col)).
    This function convert the integer to the input of the neural network.
    '''
    res = np.zeros(dim)
    s = bin(num)[::-1]
    for i in range(len(s) - 2):
        if s[i] == '1':
            res[i] = 1.0
    return res

def unit_vector(index, dim):
    '''
    convert a non-negative integer in [0, dim) to a unit vector, used in the computation of the cost.
    '''
    res = np.zeros(dim)
    res[index] = 1.0
    return res


class DataSet:
    def __init__(self, data_list, x_dim, y_dim):
        self.x = np.vstack((num_to_list(p[0], x_dim) for p in data_list))
        self.y = np.vstack((unit_vector(p[1], y_dim) for p in data_list))
        


def load_data(index):
    print('... loading data')
    with open('C:\\wingide\\NimValue\\data_sep_' + str(index) +'.pkl', 'rb') as f:
        bucket = pickle.load(f)
    x_dim = 25
    y_dim = 10
    temp = [DataSet(bucket[i], x_dim, y_dim) for i in range(3)]
    
    def shared_dataset(data_set, borrow = True):
        shared_x = theano.shared(np.asarray(data_set.x,dtype = theano.config.floatX),borrow = borrow)
        shared_y = theano.shared(np.asarray(data_set.y,dtype = theano.config.floatX),borrow = borrow)
        return shared_x, shared_y
    
    return [shared_dataset(temp[i]) for i in range(3)]

class CrossEntropy(): 
    def __init__(self,n_in,n_out):
        self.W = theano.shared(
            value = np.asarray(np.random.normal(scale = np.sqrt(1.0/n_out), size = (n_in,n_out)), dtype = theano.config.floatX),
            name = 'W',
            borrow = True        
        )
        self.b = theano.shared(
            value = np.asarray(np.random.normal(scale = 1.0, size = (n_out,)), dtype = theano.config.floatX),
            name = 'b',
            borrow =True
        )
        self.params = [self.W,self.b]
        
    def set_input(self, inpt):
        self.p_y_given_x = T.nnet.softmax(T.dot(inpt,self.W)+self.b)
        self.y_pred = T.argmax(self.p_y_given_x,axis = 1)
        
        
        
    def cross_entropy(self,y):
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x,y))


    def errors(self, y):
        a = T.argmax(y,axis = 1)
        return T.mean(T.neq(self.y_pred,a))
    
class ConvLayer():
    def __init__(self, filter_shape, image_shape, activation = T.tanh):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation = activation
        w_dim = (filter_shape[0]*filter_shape[2]*filter_shape[3])
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/w_dim), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]
    
    def set_input(self, inpt, batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = T.nnet.conv2d(input = self.inpt, filters = self.w, filter_shape = self.filter_shape, input_shape = self.image_shape)
        conv_out = T.reshape(conv_out, (batch_size, self.filter_shape[0]*(5 - self.filter_shape[3] + 1)*(5 - self.filter_shape[2] + 1)))
        self.output = self.activation(conv_out + self.b)
        
class FullyConnectedLayer():
    def __init__(self, n_in, n_out, activation = T.tanh):
        self.activation = activation
        self.w = theano.shared(
            np.asarray(np.random.normal(loc = 0.0, scale = np.sqrt(1.0/n_out), size = (n_in, n_out)), dtype = theano.config.floatX),
            name = 'w', borrow = True)
        self.b = theano.shared(
            np.asarray(np.random.normal(size = (n_out,)), dtype = theano.config.floatX),
            name = 'b', borrow = True)
        self.params = [self.w, self.b]
        
    def set_input(self, inpt):
        self.output = self.activation(T.dot(inpt, self.w) + self.b)

'''
class WrapperLayer():
    def __init__(self, n_in, n_out, batch_size, fm_num):
        self.image_size = (batch_size, 1) + n_in
        self.batch_size = batch_size
        self.n_out = n_out
        self.conv_layer = ConvLayer(filter_shape = (fm_num, 1, 2, 2), image_shape = self.image_size) 
        self.params = self.conv_layer.params
        
    def set_input(self, inpt):
        self.conv_layer.set_input(inpt, batch_size = self.batch_size)
        self.output = self.conv_layer.output
'''


class WrapperLayer():
    def __init__(self, n_in, n_out, batch_size, fm_num = 2):
        '''
        :type fm_num: int which denotes the number of feature maps.
        '''
        self.image_size = (batch_size, 1) + n_in
        self.batch_size = batch_size
        self.n_out = n_out
        self.conv_layers = [ConvLayer(filter_shape = (fm_num, 1, i, j), image_shape = self.image_size) for i in range(2, 6) for j in range(1, 6)]
        self.params = []
        for layer in self.conv_layers:
            self.params.extend(layer.params)
        
    def set_input(self, inpt):
        for layer in self.conv_layers:
            layer.set_input(inpt, batch_size = self.batch_size)
        self.output = T.concatenate([layer.output for layer in self.conv_layers], axis = 1)
        


class Network():
    def __init__(self, layers, batch_size):
        self.batch_size = batch_size
        self.layers = layers
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.dmatrix('x')
        self.y = T.dmatrix('y')
        init_layer = self.layers[0]
        init_layer.set_input(self.x)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_input(prev_layer.output)
        self.output = self.layers[-1]
        self.predict = theano.function(inputs = [self.x], outputs = self.output.y_pred)
        
    def predict(self, x):
        return self.predict(x)
    

    def verify(self, bucket_index):
        errors = []
        datasets = load_data(bucket_index)
    
        train_set_x, train_set_y = datasets[0]
        train_x = train_set_x.get_value(borrow=True, return_internal_type=True)
        n_train_batches = train_x.shape[0] //self.batch_size
        index = T.iscalar()
        #y = T.ivector()
        y = T.argmax(train_set_y[index*self.batch_size : (index+1)*self.batch_size], axis = 1)
        label = theano.function(inputs = [index], outputs = y)
        for i in range(n_train_batches):
            y_label = label(i)
            y_pred = self.predict(train_x[i*self.batch_size : (i+1)*self.batch_size])
            for (i, j) in zip(y_label, y_pred):
                if i != j:
                    errors.append((i, j))
        print('Error rate is %f for No.%i bucket.' %(len(errors)/(self.batch_size*n_train_batches), bucket_index))
        return errors

        
    def sgd(self, bucket_index, n_epochs = 200, eta = 0.02):
        '''
        :type bucket_index: int which denotes the index of the bucket whose data will be used to train the model.
        '''
        datasets = load_data(bucket_index)
        
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        
        n_train_batches = train_set_x.get_value(borrow=True, return_internal_type=True).shape[0] // self.batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True, return_internal_type=True).shape[0] // self.batch_size
        n_test_batches = test_set_x.get_value(borrow=True, return_internal_type=True).shape[0] // self.batch_size
        
        print('... building the model')
        
        index = T.lscalar()
        
        cost = self.output.cross_entropy(self.y)
        
        test_model = theano.function(
            inputs = [index],
            outputs = self.output.errors(self.y),
            givens = {
                self.x: test_set_x[index*self.batch_size: (index+1)*self.batch_size],
                self.y: test_set_y[index*self.batch_size: (index+1)*self.batch_size]
            }
        )
        
        validate_model = theano.function(
            inputs = [index],
            outputs = self.output.errors(self.y),
            givens = {
                self.x: valid_set_x[index*self.batch_size: (index+1)*self.batch_size],
                self.y: valid_set_y[index*self.batch_size: (index+1)*self.batch_size]
            }
        )
    
        gparams = [T.grad(cost,param) for param in self.params]
    
        updates = [
            (param, param - eta * gparam)
            for param, gparam in zip(self.params,gparams)
        ]
    
        train_model = theano.function(
            inputs = [index],
            outputs = self.output.errors(self.y),
            updates = updates,
            givens = {
                self.x: train_set_x[index*self.batch_size : (index+1) * self.batch_size],
                self.y: train_set_y[index*self.batch_size : (index+1) * self.batch_size]
            }
        )      
        
        print('... training the model')
        

        improvement_threshold = 0.99
        best_validation_loss = np.inf
        test_score = 0.0
        start_time = timeit.default_timer()
        done_looping = False
        epoch = 0
        while epoch < n_epochs: #and not done_looping:
            epoch += 1
            for minibatch_index in range(n_train_batches):
                train_model(minibatch_index)
            this_validation_loss = np.mean([validate_model(i) for i in range(n_valid_batches)])
            print('epoch %i, minibatch %i/%i, validation error %f %%' %(epoch, minibatch_index + 1, n_train_batches, this_validation_loss*100))
            if this_validation_loss < best_validation_loss * improvement_threshold:
                best_validation_loss = this_validation_loss
                test_score = np.mean([test_model(i) for i in range(n_test_batches)])
                print('epoch %i, minibatch %i/%i, test error of best model %f %%' %(epoch, minibatch_index + 1, n_train_batches, test_score*100))
                with open('nim_best_conv_model_2D_logistic.pkl', 'wb') as f:
                    pickle.dump(self.layers, f)
                    f.close()
                #if patience <= iter:
                    #done_looping = True
                    #break
        end_time = timeit.default_timer()
        print('Optimization complete with best validation score of %f %% with test performace %f %%' %(best_validation_loss*100, test_score*100))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.1fs' % (end_time - start_time)))
     
if __name__ == '__main__':
    layers = [FullyConnectedLayer(25, 300), FullyConnectedLayer(300, 60), CrossEntropy(60, 10)]
    #layers = pickle.load(open('C:\\Users\\Ke\\nim_best_conv_model_2D.pkl', 'rb'))
    nets = Network(layers, 50)
    nets.sgd(1) 
    
    
'''
Using two hidden layer(25, 300, 60, 10), the model achieves an error rate slightly below 10%, which is considerably worse than the result of one dimension flip game.
Also, adding the convolution layer to the model does not improve the accuracy significantly.
'''



    
    