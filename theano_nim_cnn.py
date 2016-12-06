"""

we will build convolutional neural network to train using theano

the trained model has accuracy at least 99.4% for 0/1 strings of length 20


"""





from __future__ import print_function
__docformat__ = 'restructedtext en'

import pickle
import os
import sys
import timeit

import numpy as np
from random import randint

import theano
import theano.tensor as T

from nimgame import DataSet, data_sets





def load_data(bucket_index):
    """ Loads the dataset
    
    
    : type bucket_index: int
    : param bucket_index: which bucket you are going to load
    """

    print('... loading data')
    #############
    # LOAD DATA #
    #############
    
    bucket = pickle.load(open('buckets_data/bucket_'+str(bucket_index)+'_datasets.p','rb'))
    
    train_set = bucket.train
    # Since train_set is arranged by blocks of the same nimber value, so we will make random permutation of this train_set
    for i in range(1,train_set.digits.shape[0]):
        j = randint(0,i+1)
        if j < i:
            train_set.digits[j], train_set.digits[i] = train_set.digits[i], train_set.digits[j].copy()
            train_set.labels[j], train_set.labels[i] = train_set.labels[i], train_set.labels[j].copy()
    valid_set = bucket.validation
    test_set = bucket.test
    # train_set, valid_set, test_set format: each of type DataSet, has members digits and labels
    # digits is a numpy array of 2 dimensions, where each row represent an example
    # labels is a numpy array of 2 dimensions, where the number of rows are the same as the number of rows of digits,
    # each row also represent the label of an example, the column of row i where it is 1 represents the label of example i
    
    
    
    def shared_dataset(data_xy,borrow = True):
        """ 
        Function that loads the dataset into shared variables
        """        
        data_x = data_xy.digits
        data_y = data_xy.labels
        shared_x = theano.shared(np.asarray(data_x,dtype = theano.config.floatX),borrow = borrow)
        shared_y = theano.shared(np.asarray(data_y,dtype = 'int32'),borrow = borrow)
        return shared_x, shared_y
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x,valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)
    
    rval = [(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_set_y)]
    return rval


    
class ConvLayer(object):
    def __init__(self,input, filter_shape, image_shape, batch_size, activation = T.tanh):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation = activation
        n_out = (filter_shape[0]*filter_shape[2]*filter_shape[3])
        W_values =  np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX
        )
        if activation == T.nnet.sigmoid:
            W_values *= 4
        self.W = theano.shared(value = W_values, name = 'W', borrow=True)
        b_values = np.asarray(np.random.normal(),dtype=theano.config.floatX)
        self.b = theano.shared(value = b_values, name = 'b', borrow=True)
        self.params = [self.W, self.b]
        self.input = input
        conv_out = T.nnet.conv2d(input = self.input.reshape(self.image_shape), filters = self.W, filter_shape = self.filter_shape, input_shape = self.image_shape)
        conv_out = T.reshape(conv_out,(batch_size, image_shape[3]+1-self.filter_shape[3]))
        self.output = self.activation(conv_out + self.b)
            
        
class WrapperLayer(object):
    def __init__(self, input, n_in, n_out, batch_size):
        self.input = input
        self.image_size = (batch_size, 1, 1, n_in)
        self.batch_size = batch_size
        self.n_out = n_out
        self.conv_layers = [
            ConvLayer(
                input = input, 
                filter_shape = (1, 1, 1, i), 
                image_shape = self.image_size, 
                batch_size = self.batch_size
            ) 
            for i in range(2, n_in+1)]
        self.params = []
        for layer in self.conv_layers:
            self.params.extend(layer.params)
        self.output = T.concatenate([layer.output for layer in self.conv_layers], axis = 1)
        
 


 
class HiddenLayer(object):
    """
            Typical hidden layer of a neural network: units are fully-connected and have
            sigmoidal activation function. 
            
    """    
    def __init__(self,input, n_in, n_out, W = None, b = None, activation = T.tanh):
        """        
        Weight matrix W is of shape (n_in,n_out) and the bias vector b is of shape (n_out,).
            
        NOTE : The nonlinearity used here is tanh by default
            
        Hidden unit activation is given by: tanh(dot(input,W) + b)
            
            
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
            
        :type n_in: int
        :param n_in: dimensionality of input
            
        :type n_out: int
        :param n_out: number of hidden units
            
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer        
        """
        self.input = input
        
        # initialize W and b
        if W is None:
            W_values = np.asarray(
                  np.random.normal(loc = 0.0, scale = np.sqrt(1.0/n_out),size = (n_in,n_out)),dtype = theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value = W_values, name = 'W', borrow = True)
            
        if b is None:
            b_values = np.zeros((n_out,),dtype = theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)
        self.W = W
        self.b = b
        
        
        lin_output = T.dot(input,self.W)+self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]



class CrossEntropy(object):
    """
    Multi-class Cross Entropy Class
    
    We use softmax regression to determine a class membership probability
    
    The cost or the loss of the model here is cross entropy, see 
    http://colah.github.io/posts/2015-09-Visual-Information/
    for a nice view of cross entropy
    """    
    def __init__(self,input,n_in,n_out):
        """
        Initialize the parameters of the cross entropy

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        
        """
        
        self.input = input
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value = np.zeros((n_in,n_out), dtype = theano.config.floatX),
            name = 'W',
            borrow = True        
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value = np.zeros((n_out,),dtype = theano.config.floatX),
            name = 'b',
            borrow =True
        )
        
        # symbolic expression for computing the matrix of class-membership
        # probabilities      
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)
        # symbolic description of how to compute prediction as class whose
        # probability is maximal        
        self.y_pred = T.argmax(self.p_y_given_x,axis = 1)
        # parameters of the model
        self.params = [self.W,self.b]
        
        
    
    def cross_entropy(self,y):
        """
        Return the mean of the cross entropy of the prediction probability
        of this model under a given target distribution.
        
        
        :type y: theano.tensor.TensorType
        :param y: corresponds to a matrix that gives for each example the
                 correct label
        """        
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x,y))
    

    def errors(self,y):
        """
        Return a float representing the number of errors in the minibatch
                over the total number of examples of the minibatch ; zero one
                loss over the size of the minibatch
        
        :type y: theano.tensor.TensorType
        :param y: corresponds to a matrix that gives for each example the
                          correct label
        """
                
        a = T.argmax(y,axis = 1)
        return T.mean(T.neq(self.y_pred,a))
        



class Network(object):
    """
    The whole multi layer Network Class

    Network is a feedforward artificial neural network model
    that has one wrapper layer of convolutions and one hidden layer of hidden units.
    
    Intermediate layers usually have as activation function tanh or the
    sigmoid function while the
    top layer is a softmax layer (defined here by a ``CrossEntropy``
    class).
    """    
    def __init__(self, input, n_in, n_wrapper, n_hidden, n_out, batch_size):
        """
        Initialize the parameters for the network
        

        
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                architecture (one minibatch)
        
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                which the datapoints lie
        
        :type n_wrapper: int
        :param n_wrapper: number of units in the convolution layer, in our case, we use n_in*(n_in-1)/2
        
        :type n_hidden: int
        :param n_hidden: number of hidden units
        
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                which the labels lie
        
        :type batch_size: int
        :param batch_size: number of examples in the batch
        
        """        
        self.batch_size = batch_size
        
        # we first build the WrapperLayer of convolutions
        self.wrapper_layer = WrapperLayer(
            input = input,
            n_in = n_in,
            n_out = n_wrapper,
            batch_size = batch_size
        )
        # then use the output of wrapper_layer to build the fully connected HiddenLayer
        self.hidden_layer = HiddenLayer(
            input = self.wrapper_layer.output,
            n_in = n_wrapper,
            n_out = n_hidden         
        )
        # finally use the output of hidden_layer to build CrossEntropy
        self.cross_entropy_layer = CrossEntropy(
            input = self.hidden_layer.output,
            n_in = n_hidden,
            n_out = n_out
        )
        self.input = input
        
        # parameter of this model are parameters of each layer
        self.params = self.wrapper_layer.params+self.hidden_layer.params+self.cross_entropy_layer.params



def sgd_cnn(bucket_index,retrain = False,learning_rate = 0.01, 
             n_epochs = 1000, batch_size = 20, n_in = 20, n_wrapper = 190, n_hidden = 64, n_out = 7):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type bucket_index: int
    :param bucket_index: the bucket which you will use to train
    
    
    :type retrain: boolean
    :param retrain: if True, load the trained model, if False, build the model from the beginning, default is False
    
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient


    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

   """    
    datasets = load_data(bucket_index)

    train_set_x, train_set_y = datasets[0]
    valid_set_x,valid_set_y = datasets[1]
    test_set_x,test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('...building the model')

    # allocate symbolic variables for the data
    index = T.lscalar() # index to a [mini]batch
    y = T.imatrix('y')

    if retrain:
        #load the best model and load the best validation loss for that model
        classifier, best_validation_loss = pickle.load(open('best_model_cnn.pkl'))           
        x = classifier.input
        print('previous best model hasa validation error ',best_validation_loss*100,'%')
    else:
        x = T.matrix('x')
        #build the Network
        classifier = Network(
            input = x,
            n_in = n_in,
            n_wrapper = n_wrapper,
            n_hidden = n_hidden,
            n_out = n_out,
            batch_size = batch_size
        )
        best_validation_loss = np.inf
        
    # the cost we minimize during training is the cross entropy of
    # the model; cost is expressed here symbolically    
    cost = classifier.cross_entropy_layer.cross_entropy(y)
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams    
    gparams = [T.grad(cost,param) for param in classifier.params]
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs    
    updates = [
        (param, param - learning_rate * gparam)
        for param,gparam in zip(classifier.params,gparams)
    ]


    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates = updates,
        givens = {
            x: train_set_x[index*batch_size : (index+1) * batch_size],
            y: train_set_y[index*batch_size : (index+1) * batch_size]
        }
    )
    
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch    
    test_model = theano.function(
        inputs = [index],
        outputs = classifier.cross_entropy_layer.errors(y),
        givens = {
            x: test_set_x[index*batch_size: (index+1)*batch_size],
            y: test_set_y[index*batch_size: (index+1)*batch_size]
        }
    )

    validate_model = theano.function(
        inputs = [index],
        outputs = classifier.cross_entropy_layer.errors(y),
        givens = {
            x: valid_set_x[index*batch_size: (index+1)*batch_size],
            y: valid_set_y[index*batch_size: (index+1)*batch_size]
        }
    )    


    ###############
    # TRAIN MODEL #
    ###############
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()    
    validation_frequency = n_train_batches
    epoch = 0
    print("....training the model")
    while epoch < n_epochs:
        epoch = epoch+1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch-1)*n_train_batches+minibatch_index
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f % %'
                    %(
                        epoch,
                        minibatch_index+1,
                        n_train_batches,
                        this_validation_loss*100.
                    )

                )
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    with open('best_model_cnn.pkl', 'wb') as f:
                        pickle.dump((classifier,this_validation_loss),f)                  
                    print(
                        'epoch %i, minibatch %i/%i, test error of best model %f % %'
                        %(
                            epoch,
                            minibatch_index+1,
                            n_train_batches,
                            test_score * 100.

                        )
                    )

    end_time = timeit.default_timer()
    print (
        'Optimization complete. Best validation score of %f % % obtained at iteration %i, with test performance %f % %'
        %(
            best_validation_loss * 100.,
            best_iter + 1,
            test_score * 100.
        )
    )

    print(
        'The code for file '+os.path.split(__file__)[1]+ ' ran for %.2fm' % ( (end_time - start_time)/60.),file = sys.stderr
    )    




def predict(bucket_index):
    """
    :type bucket_index:int
    :param bucket_index: the bucket which you will use to predict the nimber value
    
    """
    # load the best model
    classifier, error = pickle.load(open('best_model_cnn.pkl'))
    batch_size = classifier.batch_size
    x = classifier.input
    
    # compile a predictor function
    predict_model = theano.function(
        inputs = [x],
        outputs = classifier.cross_entropy_layer.y_pred
    )
    datasets = load_data(bucket_index)
    x_values, y_values = datasets[0]
    x_values = x_values.get_value(borrow = True)
    # y_values is a 1d vector which stores the real nimber value
    y_values = np.argmax(y_values.get_value(borrow = True),axis = 1)
    n_batches = x_values.shape[0] // batch_size
    
    # y_predict is a 1d vector which stores the predicted nimber value
    y_predict = np.asarray([],dtype = 'int32')
    for minibatch_index in range(n_batches):
        temp = predict_model(x_values[minibatch_index*batch_size: (minibatch_index+1)*batch_size])
        y_predict = np.concatenate([y_predict,temp])
        
    predict_error = np.mean(np.not_equal(y_predict,y_values[:y_predict.shape[0]]))
    print("prediction error on these ", y_predict.shape[0], " cases is: ", predict_error*100,"%")
    return predict_error*100




                
if __name__ == '__main__':
    x = T.matrix('x')
    classifier = Network(
            input = x,
            n_in = 20,
            n_wrapper = 190,
            n_hidden = 64,
            n_out = 7,
            batch_size = 20
        )
    pickle.dump(classifier,open('test.pkl','wb'))