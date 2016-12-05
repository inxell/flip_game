from collections import defaultdict
from collections import Counter
import random
import pickle


class FlipGame2D:
    def __init__(self, row, col):
        self.dim = row*col
        self.row = row
        self.col = col
        self.nim_vals = [None]*(2**self.dim)
        '''
        self.nim_vals[i] denote the nim-value of the configuration corresponding to integer i.
        Here integer i corresponds to the configuration bin(i)[2:].zfill(self.dim)[::-1].reshape((row, col)).
        '''
        self.nim_vals[0] = 0
        for i in range(self.dim):
            self.nim_vals[2**i] = 0
        for i in range(row):
            for j in range(col-1):
                index = 2**(i*col + j) + 2**(i*col + j + 1)
                self.nim_vals[index] = 1
        for j in range(col):
            for i in range(row -1):
                index = 2**(i*col + j) + 2**((i+1)*col + j)
                self.nim_vals[index] = 1
        for i in range(1, self.dim):
            for j in range(i):
                index = 2**i + 2**j
                if self.nim_vals[index] is None:
                    self.nim_vals[index] = 0
                    
    def nimval(self, n):
        if self.nim_vals[n] is not None:
            return self.nim_vals[n]
        nbin = bin(n)[2:][::-1]
        next_state_vals = set()
        for i in range(len(nbin)-1):
            if (i+1)%self.col != 0 and nbin[i:i+2] == '11':
                next_state = n - 2**i - 2**(i+1)
                next_state_vals.add(self.nim_vals[next_state])
        for i in range(len(nbin) - self.col):
            if nbin[i] == '1' and nbin[i+self.col] == '1':
                next_state = n - 2**i - 2**(i+self.col)
                next_state_vals.add(self.nim_vals[next_state])
        for i in range(self.dim): # mex of the nim-values of all next states.
            if i not in next_state_vals:
                self.nim_vals[n] = i
                break
        return self.nim_vals[n]
 

    def update(self, file_addr = None):
        '''
        :type file_addr: string which denotes the address of txt file where we want to store the results.
        If file_addr is None, we do not store the result in the disk.
        '''
        def recurGen(g):
            for s in g:
                for d in '01':
                    yield s + d
        g = ['']
        for i in range(self.dim):
            g = recurGen(g)
        # g is the generator which generates all '01' string of length self.dim in increasing order.
        if file_addr is None:
            for i in range(2**self.dim):
                self.nimval(i)
                if i%100000 == 0:
                    print(i)
            return
        f = open(file_addr, 'a')
        for i in range(2**self.dim):
            f.write(next(g) + '    ' + str(self.nimval(i)) + '\n')
            if i%100000 == 0:
                f.flush()
                print(i)
        f.close()


fg2d = FlipGame2D(5, 5)
fg2d.update()
#fg2d.update('C:\\wingide\\NimValue\\NimVals2D.txt')
pickle.dump(fg2d.nim_vals, open('C:\\wingide\\NimValue\\NimVals2D.pkl', 'wb'))
counter = Counter(fg2d.nim_vals)
print(counter)
print(sum(counter.values()))

# separate the data into 10 bucket with relatively equal size.
whole_set = pickle.load(open('C:\\wingide\\NimValue\\NimVals2D.pkl', 'rb'))
data_sets = [[] for i in range(10)]
for i in range(len(whole_set)):
    index = random.randint(0, 9)
    data_sets[index].append((i, whole_set[i]))
for i in range(10):
    with open('C:\\wingide\\NimValue\\data_set_' + str(i) +'.pkl', 'wb') as f:
        pickle.dump(data_sets[i], f)
        f.close()

total = Counter([])
for i in range(10):
    bucket = pickle.load(open('C:\\wingide\\NimValue\\data_set_' + str(i) +'.pkl', 'rb'))
    cnt = Counter([p[1] for p in bucket])
    total += cnt
    print('Bucket ' + str(i) +':')
    print(cnt)
    print('total data points in bucket_' + str(i) + ' is %d' %len(data_sets[i]))
    print([cnt[i]/cnt[0] for i in range(9)])



def bucket_sep(addr, index):
    '''
    :type addr: string which donotes the pickled file of bucket.
    This function separate the data in the bucket into [train, validation, test] three partition with approximate ratio 7:1:2, 
    and dump the separated data into an pkl file with index in the file name.
    '''
    bucket = pickle.load(open(addr, 'rb'))
    train, validation, test = [], [], []
    for p in bucket:
        i = random.randint(0, 9)
        if i < 7:
            train.append(p)
        elif i == 7:
            validation.append(p)
        else:
            test.append(p)
    random.shuffle(train)
    with open('C:\\wingide\\NimValue\\data_sep_' + str(index) +'.pkl', 'wb') as f:
        pickle.dump([train, validation, test], f)
        f.close()
        
        
for i in range(3):
    print(i)
    addr = 'C:\\wingide\\NimValue\\buckets\\data_set_' + str(i) + '.pkl'
    bucket_sep(addr, i)



                
        
        
        
        