import numpy as np

class Neural_Network(object):
    def __init__(self): #design of Network
#parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

#weighs
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3)
    #print(self.W1)
        self.W2 = np.random.randn(self.hiddenSize,self.outputSize) #(3x1)
    #print(self.W2)    
    
    def forward(self,X): #forward propagation through our network
        self.z = np.dot(X,self.W1)         #dot product of X (input) and 
        self.z2 = self.sigmoid(self.z)     #activation function
        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden 
        o = self.sigmoid(self.z3)          #final activation function
        return o
    
    def sigmoid(self,s):           #activation function
        return 1/(1+np.exp(-s))
    
    def sigmoidPrime(self,s):     #derivative of sigmoid
        return s * (1 - s)
    
    def backward (self, X, y, o): #backward propagate through the network
        
        self.o_error = y - o       #error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) #applying derivative of sigmoid to 
        
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
        
        
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)#here comparing with forward additionally I need to output because our approch is bachword  
        
    def saveWeights(self):
        np.savetxt("w1.txt",self.W1, fmt="%s") 
        np.savetxt("w2.txt" ,self.W2, fmt = "%s")
        
    def predict(self):
       print("predicted data based on trained weights: ")
       print("Input (scaled): \n" + str(Xpredicted))
       print("output: \n" + str(self.forward(Xpredicted)))



# X = (hours studing, hours sleeping), y = score on test, Xpredicted = 4 hours & studing 8 hours
X = np.array(([2,9], [1,5], [3,6]),dtype = float )
y = np.array(([92],[86],[89]), dtype = float)
Xpredicted = np.array(([4,8]),dtype = float)

#scale units
##print(x)
X = X/np.amax(X,axis = 0) #maximum of x array
## max_of(2,1,3) = 3 ==> array will be as (2/3, 1/3, 3/3)
## max_of(9,5,6) = 9 ==> array will be as (9/9, 5/9, 6/9)
##print(X)

##also the scaling will be for Xpredicted
##print(Xpredicted)
Xpredicted = Xpredicted/ np.amax(Xpredicted,axis=0) 
##print(Xpredicted)

#we divided by 100 because we have no idea yet about the output so it is possible to be higher the available values
y = y/100


NN = Neural_Network()

for i in range(500): #trains the NN 1000 times
    print("#" + str(i) + "\n")
    print ("input (scaled): \n" + str(X))
    print ("actual Output: \n " + str(y))
    print ("predicted output: \n" + str(NN.forward(X)))
    print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))))
    print ("\n")
    
    NN.train(X, y)


NN.saveWeights()
NN.predict()



