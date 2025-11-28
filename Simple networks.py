import numpy as np
from scipy import signal

# Functions

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

class Conv4d:
    def __init__(self, kernel, channels, image_shape, filters=1):
        self.channels = channels
        self.filters = filters
        self.output_shape = (filters, image_shape[0] - kernel + 1, image_shape[1] - kernel + 1)
        self.kernels_shape = (filters, channels, kernel, kernel)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
    def forward(self,X,batch=1):
        self.input = X
        self.output = np.copy(self.biases)
        if batch >= 0:
            outputs = []
            for n in range(batch):
                for i in range(self.filters):
                    for j in range(self.channels):
                        #print(self.input[n][j].shape, self.kernels[i, j].shape)
                        self.output[i] += signal.correlate2d(self.input[n][j], self.kernels[i, j], "valid")
                outputs.append(self.output)
                self.output = np.copy(self.biases)
            outputs = np.array(outputs)
            return outputs
        else:
            for i in range(self.filters):
                for j in range(self.channels):
                    self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
            return self.output
    def backward(self,output_gradient,learning_rate=0.01,batch=1,auto_grad_return=True):
        if auto_grad_return != False:
            kernels_gradient = np.zeros(self.kernels_shape)
            input_gradient = np.zeros(self.input.shape)
            for n in range(batch):
                for i in range(self.filters):
                    for j in range(self.channels):
                        kernels_gradient[i,j] += signal.correlate(in1=output_gradient[n,i], in2=self.input[n,j], mode="valid")
                        input_gradient[n,j] += signal.convolve(output_gradient[n,i], self.kernels[n,j], "full")
            self.kernels -= learning_rate * kernels_gradient
            return input_gradient
        else:
            kernels_gradient = np.zeros(self.kernels_shape)
            input_gradient = np.zeros(self.input.shape)
            for n in range(batch):
                for i in range(self.filters):
                    for j in range(self.channels):
                        kernels_gradient[i,j] += signal.correlate(in1=output_gradient[n,i], in2=self.input[n,j], mode="valid")
                        input_gradient[n,j] += signal.convolve(output_gradient[n,i], self.kernels[n,j], "full")
            return kernels_gradient, input_gradient
    def manual_update_kernel(self,k_grad,lr=0.01):
        self.kernels -= lr * k_grad

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

def reshape4d(IN,dim=1):
    total = 1
    for n in IN.shape[dim:]:
        total*=n
    return np.reshape(IN,(IN.shape[0],total))

def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)))

def sigmoid_derivative(x):
    x = 1/(1 + np.exp(-x))
    return x * (1 - x)

def ReLU(x):
    return x * (x > 0)

def derivative_ReLu(x):
    return 1 * (x > 0)
        
def batch_list(input_list, batch_size):
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]



# Loading data I downloaded my mnist of the internet and process locally but I'm sure you could probably just load a mnist out of a library

train_file = open("mnist_train.csv")
train_list = train_file.readlines()
train_file.close()


test_file = open("mnist_test.csv")
test_list = test_file.readlines()
test_file.close()

batch_size = 8

Batch_list = np.array(batch_list(train_list,batch_size))
lr = 0.01
pool=MaxPooling4D()


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
        
# Standard MultiLayerPerceptron for MNIST 
"""MLP = MultiLayerPerceptron(196,128,64,10)
for E in range(10):
    predictions = []
    for batch in Batch_list:
        # Embedding Layer
        n = [arr.split(',') for arr in batch]
        X = [arr[1:] for arr in n]
        X = np.asfarray(X)
        Y = np.zeros((batch_size,10)) + 0.01
        X = np.reshape(X, (batch_size,1,28,28))
        X = pool.forward(X,(2,2),2)
        X = np.reshape(X, (batch_size,196))
        X = X/255*0.99+0.01
        for i, value in enumerate(Y):
            Y[i][int(n[i][0])]=0.99
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = Y[indices]
        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
        Z = MLP.forward(X_batch)
        loss = 2*(Z-y_batch)/10
        MLP.backward(loss,lr)
        pred = [np.argmax(i) for i in Z]
        target = [np.argmax(i) for i in y_batch]
        for i in range(batch_size):
            predictions.append(pred[i]==target[i])
    print(E, " ", np.mean(predictions)*100)"""



class SimpleConv:
    def __init__(self):
        self.Conv1 = Conv4d(3,1,[28,28],6)
        self.Conv2 = Conv4d(3,6,[13,13],8)
        self.pool1 = MaxPooling4D()
        self.pool2 = MaxPooling4D()
        self.Lin1 = Linear(200,128)
        self.Lin2 = Linear(128,64)
        self.Lin3 = Linear(64,10)
    def forward(self,X):
        self.A0 = X
        self.GA1 = self.pool1.forward(self.Conv1.forward(X,8),stride=2)
        self.A1 = ReLU(self.GA1)
        self.GA2 = self.pool2.forward(self.Conv2.forward(self.A1,8),stride=2)
        self.A2 = ReLU(self.GA2)
        self.PA2 = reshape4d(self.A2,1)
        self.GA3 = self.Lin1.forward(self.PA2)
        self.A3 = sigmoid(self.GA3)
        self.GA4 = self.Lin2.forward(self.A3)
        self.A4 = sigmoid(self.GA4)
        self.GA5 = self.Lin3.forward(self.A4)
        self.A5 = sigmoid(self.GA5)
        return self.A5
    def backward(self,y,lr=0.1):
        Loss=2*(self.A5-y)/10 * sigmoid_derivative(self.GA5)
        Loss,DW3,DB3 = self.Lin3.backward(Loss)
        Loss *= sigmoid_derivative(self.GA4)
        Loss,DW2,DB2 = self.Lin2.backward(Loss)
        Loss *= sigmoid_derivative(self.GA3)
        Loss,DW1,DB1 = self.Lin1.backward(Loss)
        Loss = np.reshape(Loss,(8,8,5,5))
        Loss *= derivative_ReLu(self.GA2)
        Loss = self.pool2.backprop(Loss)
        Loss = self.Conv2.backward(Loss,lr)
        Loss *= derivative_ReLu(self.GA1)
        Loss = self.pool1.backprop(Loss)
        self.Conv1.backward(Loss,lr)
        self.Lin1.Update(DW1,DB1,lr)
        self.Lin2.Update(DW2,DB2,lr)
        self.Lin3.Update(DW3,DB3,lr)

# Standard Convultion with Linear layers
"""conv = SimpleConv()
for E in range(10):
    predictions = []
    for batch in Batch_list:
        # Embedding Layer
        n = [arr.split(',') for arr in batch]
        X = [arr[1:] for arr in n]
        X = np.asfarray(X)
        Y = np.zeros((batch_size,10)) + 0.01
        X = X/255*0.99+0.01
        for i, value in enumerate(Y):
            Y[i][int(n[i][0])]=0.99
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = Y[indices]
        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
        X_batch = np.reshape(X_batch, (batch_size,1,28,28))
        Z = conv.forward(X_batch)
        conv.backward(y_batch,lr)
        pred = [np.argmax(i) for i in Z]
        target = [np.argmax(i) for i in y_batch]
        for i in range(batch_size):
            predictions.append(pred[i]==target[i])
    print(E, " ", np.mean(predictions)*100)"""



def nt_xent_loss(embeddings, temperature=0.5):
    # Normalize embeddings
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    
    # Extract positive pairs (assumes positive pairs are organized sequentially)
    N = embeddings.shape[0] // 2
    positive_pairs = np.array([[i, i + N] for i in range(N)] + [[i + N, i] for i in range(N)])
    
    # Compute loss
    loss = 0
    for i, j in positive_pairs:
        numerator = np.exp(similarity_matrix[i, j] / temperature)
        denominator = np.sum(np.exp(similarity_matrix[i, :] / temperature)) - numerator
        loss += -np.log(numerator / denominator)
    
    return (loss+0.00001) / (2 * N+0.00001)
class SimpleContrastive:
    def __init__(self):
        self.Lin1 = Linear(784,128)
        self.Lin2 = Linear(128,64)
        self.Lin3 = Linear(64,10)
    def forward(self,X):
        self.A0 = X
        self.GA1 = self.Lin1.forward(self.A0)
        self.A1 = sigmoid(self.GA1)
        self.GA2 = self.Lin2.forward(self.A1)
        self.A2 = sigmoid(self.GA2)
        self.GA3 = self.Lin3.forward(self.A2)
        self.A3 = sigmoid(self.GA3)
        return self.A3
    def backward(self,Z1,Z2,lr=0.1,t=0.7):
        Loss = nt_xent_loss(np.concatenate((Z1,Z2)),t) * sigmoid_derivative(self.GA3)
        Loss,DW3,DB3 = self.Lin3.backward(Loss)
        Loss *= sigmoid_derivative(self.GA2)
        Loss,DW2,DB2 = self.Lin2.backward(Loss)
        Loss *= sigmoid_derivative(self.GA1)
        Loss,DW1,DB1 = self.Lin1.backward(Loss)
        self.Lin1.Update(DW1,DB1)
        self.Lin2.Update(DW2,DB2)
        self.Lin3.Update(DW3,DB3)

# Contrastive model I say a failure but does work
"""
net = SimpleContrastive()
Batch_list = np.array(batch_list(train_list,batch_size))
for E in range(20):
    predictions = []
    t=1.5
    for batch in Batch_list[:1000]:
        n = [arr.split(',') for arr in batch]
        X = [arr[1:] for arr in n]
        X2 =[arr[1:] for arr in n]
        X = np.asfarray(X)/255*0.99+0.01
        X2 = ((np.asfarray(X2)-255)*-1)/255*0.99+0.01
        Y = np.zeros((batch_size,10)) + 0.01
        for i, value in enumerate(Y):
            Y[i][int(n[i][0])]=0.99
        Z = net.forward(X)
        Z2 = net.forward(X2)
        net.backward(Z,Z2,t=t)
        pred = [np.argmax(i) for i in Z]
        target = [np.argmax(i) for i in Y]
        for i in range(batch_size):
            predictions.append(pred[i]==target[i])
    print(E, " ", np.mean(predictions)*100)"""