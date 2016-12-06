
"""

 You and your friend are playing the following game:
     you are given a 0/1 string of length n
     you could flip two consecutive 1's to two 0's
     the one who could not find two consecutive 1's lose the game
 This game is equivalent to the Nimber Game (Sprague-Grundy theorem, 
 see notes on this theorem http://udel.edu/~shuying/nimgame.pdf)
 Each 0/1 string has its own nimber value, the first player could win 
 if and only if the corresponding nimber value is not 0. 

 For example, '111' has nimber value 1, whereas '101' has nimber value 0.


 Our goal for this project is to train the nimber value of 0/1 string of length 20.
 Notice that this will also give you the nimber value of 0/1 string of length < 20.
(since for a string of length n < 20, it could be thought of as a string of length 20, 
with the last 20-n positions are all 0, e.g. for '11001111', it is the same as 
'11001111000000000000')
 
 
 
 
 nimgame.py is to preprocess the data we need to train the model
 
 
 
"""

from __future__ import print_function
__docformat__ = 'restructedtext en'
from random import randint
import pickle
import numpy as np






class NimGame:
    """
    NimGame Class 
    
    for each 0/1 string of length n, we find its nimber value by using dynamic programming
    
    """
    
    def __init__(self,n):
        """
        : type n: int    
        : param n: length of the string
        
        """
        self.length = n
        
        # initialize the dp list
        self.dp = [-1]*(n+1)
        
    def findNimConsecutive(self,l):
        """
        : type l: int
        : param l: the number of consecutive 1's
        """
        if self.dp[l] >= 0:
            return self.dp[l]
        if l == 0 or l == 1:
            self.dp[l] = 0
            return 0
        appeared = set()
        for j in range(1,l):
            appeared.add(self.findNimConsecutive(j-1)^self.findNimConsecutive(l-j-1))
        res = 0
        while res in appeared:
            res += 1
        self.dp[l] = res
        return res
    
    def nim_value(self,s):
        """
        : type s: str
        : param s : the string s we are going to find its nimber value
        """
        res = 0
        i = 0
        while i < len(s):
            if s[i] == '1':
                start = i
                while i < len(s) and s[i] == '1':
                    i += 1            
                res = res ^ self.findNimConsecutive(i-start)
            i += 1         
        return res
    
    def random_input(self):
        """
        generate a random 0/1 string, each 0/1 string can be also viewed as a binary representation 
        of an integer
        
        """
        s = random.randint(0,2**self.length-1)
        return format(s,'0'+str(self.length)+'b')




     
class DataSet:
    """
    DataSet Class is to store the input matrix and the output matrix   
    
    """
    def __init__ (self,inputs,n,max_label):
        """
        : type inputs: list
        : param inputs: inputs is a list of [string,its_nimber_value]
        
        
        : type n: int
        : param n: length of the string
        
        
        : type max_label: int
        : param max_label: the largest nimber value + 1, (if length of the string is 20, then this max_label is 7)
        """
        
        # initialize input matrix (digits), which is a numpy array        
        self.digits = np.empty([0,n])
        # initialzie output matrix (labels), which is a numpy array
        self.labels = np.empty([0,max_label])
        
        """
         for each element in the inputs list, process it and add it to the digits and labels
         e.g. for n = 20, and string '11100000000000000000'(its nimber value is 3)
         we add one row [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] to the digits
         and we add one row [0,0,0,1,0,0,0] to the labels
        """
        for p in inputs:
            d = p[0]
            label = p[1]
            l = list()
            for c in d:
                if c == '1':
                    l.append(1)
                else:
                    l.append(0)
            l = np.array(l)
            self.digits = np.vstack((self.digits,l))
            r = np.zeros(max_label)
            r[label] = 1
            self.labels = np.vstack((self.labels,r))


class data_sets:
    """
    data_sets Class is to randomly split the data into train data, validation data and test data
    
    |test| : |validation| + |train| = 1 : 5
    
    |validation| : |train| = 1 : 10
    
    when we split the data, we do it for strings with nimber value 0 using this proportion, and then for strings with nimber
    value 1 using this proportion, and so on.
    
    """
    def __init__(self,n,NV):  
        """        
        : type n: int
        : param n: length of the string
    
        : type NV: dict
        : param NV: maps nimvalue to a list of string which has this nimber value
        """
        test = list()
        train = list()
        validation = list()
        for key in NV:
            print('this is for nimber value ',key)
            digits = NV[key]
            length = len(digits)
            
            # number of test data we need for this category
            test_length_goal = max(length/6,1)
            
            # number of validation data we need for this category
            validation_length_goal = max((length-test_length_goal)/11,1)
            
            # number of test data before we process this category
            test_length_cur = len(test)
            
            # number of validation data before we process this category
            validation_length_cur = len(validation)
            
            
            # use Reservoir sampling 
            for i in range(len(digits)):
                cur = [digits[i],key]
                if len(test) < test_length_cur+test_length_goal:
                    test.append(cur)
                else:
                    j = random.randint(0,i+1)
                    if j < test_length_goal:
                        temp = test[test_length_cur+j]
                        test[test_length_cur+j] = cur
                        cur = temp
                    if len(validation) < validation_length_cur+validation_length_goal:
                        validation.append(cur)
                    else:
                        k = random.randint(0,i+1-test_length_goal)
                        if k < validation_length_goal:
                            temp = validation[validation_length_cur+k]
                            validation[validation_length_cur+k] = cur
                            cur = temp
                        train.append(cur)
        self.test = DataSet(test,n,len(NV))
        print("done test")
        self.validation = DataSet(validation,n,len(NV))
        print("done validation")
        self.train = DataSet(train,n,len(NV))
        print("done train")
        
        

class bucket_data_set:
    """
    
    Divide the 2^n strings into 10 buckets using Reservoir sampling 
    
    we will use train set of one bucket to train the model
    
    
    """
    def __init__(self,n):
        """
        : type n: int
        : param n: length of the string
        """       
        
        NV = pickle.load(open('nim.p','rb'))
        self.bucket = [dict() for i in range(10)]
        for key in NV:
            for i in range(10):
                self.bucket[i][key]  = list()
            digits = NV[key]
            length = len(digits)
            goal = length/10+1
            for i in range(length):
                a = i/goal
                if a == 0:
                    self.bucket[0][key].append(digits[i])
                else:
                    cur = digits[i]
                    k = random.randint(0,i+1)
                    b = k/goal
                    c = k%goal
                    if b < a:
                        temp = self.bucket[b][key][c]
                        self.bucket[b][key][c] = cur
                        cur = temp
                    self.bucket[a][key].append(cur)
        self.bucket_data = []
        for i in range(10):
            print("doing for batch ",str(i))
            self.bucket_data.append(data_sets(n,self.bucket[i]))
                    
                        




        
        






        
    