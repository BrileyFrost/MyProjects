import numpy as np
from scipy.special import erf

# Standard Functions
def Softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)))

def sigmoid_derivative(x):
    x = 1/(1 + np.exp(-x))
    return x * (1 - x)

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_derivative(x):
    phi_x = 0.5 * (1 + erf(x / np.sqrt(2)))
    dphi_dx = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    return phi_x + x * dphi_dx


# Networks
class Linear:
    def __init__(self,In,Out,ones=False):
        if ones != False:
            self.W = np.zeros((In,Out))
        else:
            self.W = np.random.randn(In,Out)* np.sqrt(1. / Out)
        self.bias = np.zeros(Out)
    def forward(self,X):
        self.X = X
        return np.dot(X,self.W)+self.bias
    def backward(self,loss):
        error  = np.dot(loss,self.W.T)
        grad_weights = np.dot(self.X.T,loss)
        grad_bias = np.sum(loss, axis=0)
        return error,grad_weights,grad_bias
    def Update(self,grad_weights,grad_bias,learning_rate=0.001):
        self.W -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

class GeluMultiLayerPerceptron():
    def __init__(self,In,H,H1,Out,ones=False):
        self.L1 = Linear(In,H,ones)
        self.L2 = Linear(H,H1,ones)
        self.L3 = Linear(H1,Out,ones)
    def forward(self,X):
        self.X0 = X
        self.X1 = self.L1.forward(X)
        self.A1 = gelu(self.X1)
        self.X2 = self.L2.forward(self.X1)
        self.A2 = gelu(self.X2)
        self.X3 = self.L3.forward(self.X2)
        self.A3 = sigmoid(self.X3)
        return self.A3
    def backward(self,error,lr=0.001,return_error=False):
        error *= gelu_derivative(self.X3)
        error,Dw3,Db3=self.L3.backward(error)
        error *= gelu_derivative(self.X2)
        error,Dw2,Db2=self.L2.backward(error)
        error *= sigmoid_derivative(self.X1)
        error,Dw1,Db1=self.L1.backward(error)
        self.L1.Update(Dw1,Db1,lr)
        self.L2.Update(Dw2,Db2,lr)
        self.L3.Update(Dw3,Db3,lr)
        if return_error != False:
            return error

class Attention_Block:
    def __init__(self,LinIn,LinH1,LinH2,LinOut,Dimensions,Mask=False):
        self.Mask = Mask
        self.q = Linear(Dimensions,Dimensions)
        self.k = Linear(Dimensions,Dimensions)
        self.v = Linear(Dimensions,Dimensions)
        self.D=Dimensions
        self.lin = GeluMultiLayerPerceptron(LinIn,LinH1,LinH2,LinOut)
    def Embed(self,X):
        q = self.q.forward(X)
        k = self.k.forward(X)
        v = self.v.forward(X)
        return q,k,v
    def ScaledDotProduct(self,Q,K,V):
        Z=np.matmul(Q,K.T)/self.D
        if self.Mask==True:
            pass
        Z=np.matmul(Softmax(Z),V)
        return Z
    def SumNorm(self,X):
        Z=X#BatchNorm(X)
        return Z
    def FeedForward(self,X):
        self.Z=self.lin.forward(X)
        return self.Z
    def ForwardPass(self,In,N_p):
        Zs = []
        for i in range(N_p):
            q,k,v = self.Embed(In[i])
            self.Attention = self.ScaledDotProduct(q,k,v)
            Z = self.SumNorm(self.Attention)
            Zs.append(Z)
        Np = N_p//2
        self.Np = N_p
        for i in range(Np):
            Zs=np.concatenate((Zs[i],Zs[i+Np]),axis=1)
        Z = self.SumNorm(self.FeedForward(Zs))
        return Z
    def Backward(self,Error,lr=0.001):
        Error = self.lin.backward(Error,lr,return_error=True)
        Error=np.array(np.hsplit(Error,self.Np))
        for error in Error:
            _, Dw, Db = self.q.backward(error)
            self.q.Update(Dw,Db,lr)
            _, Dw, Db = self.k.backward(error)
            self.k.Update(Dw,Db,lr)
            _, Dw, Db = self.v.backward(error)
            self.v.Update(Dw,Db,lr)

class MultiLayerPerceptron():
    def __init__(self,In,H,H1,Out,ones=False):
        self.L1 = Linear(In,H,ones)
        self.L2 = Linear(H,H1,ones)
        self.L3 = Linear(H1,Out,ones)
    def forward(self,X):
        self.X0 = X
        self.X1 = self.L1.forward(X)
        self.A1 = sigmoid(self.X1)
        self.X2 = self.L2.forward(self.X1)
        self.A2 = sigmoid(self.X2)
        self.X3 = self.L3.forward(self.X2)
        self.A3 = sigmoid(self.X3)
        return self.A3
        #return log_softmax(self.A3)
    def backward(self,error,lr=0.001,return_error=False):
        error *= sigmoid_derivative(self.X3)
        error,Dw3,Db3=self.L3.backward(error)
        error *= sigmoid_derivative(self.X2)
        error,Dw2,Db2=self.L2.backward(error)
        error *= sigmoid_derivative(self.X1)
        error,Dw1,Db1=self.L1.backward(error)
        self.L1.Update(Dw1,Db1,lr)
        self.L2.Update(Dw2,Db2,lr)
        self.L3.Update(Dw3,Db3,lr)
        if return_error != False:
            return error

class MaxPooling4D:
    def __init__(self):
        pass
    def forward(self,Image,pool_size=(2,2),stride=1):
        batch_size, channels, height, width = Image.shape
        self.input_shape=Image.shape
        pool_h, pool_w = pool_size
        
        # Output dimensions
        out_h = (height - pool_h) // stride + 1
        out_w = (width - pool_w) // stride + 1
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        window = Image[b, c, 
                                            i*stride:i*stride+pool_h, 
                                            j*stride:j*stride+pool_w]
                        max_index = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indices[b, c, i, j] = max_index
                        output[b, c, i, j] = window[max_index]
    
        return output
    
    def backprop(self, dout, stride=1):
        batch_size, channels, out_h, out_w = dout.shape
        grad_input = np.zeros(self.input_shape)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        i_idx, j_idx = self.max_indices[b, c, i, j]
                        grad_input[b, c, i*stride+i_idx, j*stride+j_idx] += dout[b, c, i, j]
        
        return grad_input


# Data prep
def batch_list(input_list, batch_size):
    """Divides a list into smaller sub-lists of a fixed size."""
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


train_file = open("mnist_train.csv")
train_list = train_file.readlines()
train_file.close()


test_file = open("mnist_test.csv")
test_list = test_file.readlines()
test_file.close()

batch_size = 8
N_Patches = 2

Batch_list = np.array(batch_list(train_list,batch_size))
lr = 0.001

pool = MaxPooling4D()
Decoder = Attention_Block(196,128,128,98,98)
mlp = MultiLayerPerceptron(98,64,32,10)
for E in range(10):
    predictions = []
    for batch in Batch_list:
        # Embedding Layer
        n = [arr.split(',') for arr in batch]
        X = [arr[1:] for arr in n]
        X = np.asfarray(X)
        Y = np.zeros((batch_size,10)) + 0.01
        for i, value in enumerate(Y):
            Y[i][int(n[i][0])]=0.99
        X = np.reshape(X, (batch_size,1,28,28))
        X = pool.forward(X,(2,2),2)
        X = np.reshape(X, (batch_size,196))
        X = X/255*0.99+0.01
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = Y[indices]
        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
        X_batch = np.array(np.hsplit(X_batch,N_Patches))
        Z = Decoder.ForwardPass(X_batch, N_Patches)
        Z = mlp.forward(Z)
        Error = 2*(Z-y_batch)/10
        Error = mlp.backward(Error, lr, True)
        Decoder.Backward(Error,lr)
        pred = [np.argmax(i) for i in Z]
        target = [np.argmax(i) for i in y_batch]
        for i in range(batch_size):
            predictions.append(pred[i]==target[i])
    print(E, " ", np.mean(predictions)*100)