import torch
import math
import numpy as np
from scipy.special import expit as sigmoid

def rnn(wt_h, wt_x, bias, init_state, input_data):
    """
    RNN forward calculation.

    args:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation. Rows corresponds 
              to dimensions of previous hidden states
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term
        init_state: shape [hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    returns:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """
    '''for X in inputs:
      state = torch.tanh(torch.matmul(torch.from_numpy(X),wt_x) + (torch.matmul(state, wt_h) if state is not None else 0) + bias)
      outputs.append(state)'''

    batches = input_data.shape[0]
    outputs = np.empty((batches, 0)).tolist()


    state = None
    if init_state is not None:
      state, = init_state
    inputs = np.transpose(input_data,(1,0,2))
    inputs = torch.from_numpy(inputs)
    wt_x = torch.from_numpy(wt_x)
    wt_h = torch.from_numpy(wt_h)
    bias = torch.from_numpy(bias)
    state = torch.from_numpy(state)
    i = 0
    for X in inputs:
      state = torch.tanh(torch.matmul(X, wt_x) + (
            torch.matmul(state, wt_h) if state is not None else 0)
                         + bias)
      if i == 0:
        i+=1
        for index, s in enumerate(state):
          s = torch.unsqueeze(s,0)                   
          outputs[index] = s
      else:
        for index, s in enumerate(state):
          n = outputs[index]
          s = torch.unsqueeze(s,0)
          m = torch.cat([n,s])
          outputs[index] = m
    outputs = torch.stack(outputs)
    outputs = outputs.numpy()
    state = state.numpy()


    return outputs, state


def gru(linear_trans_r, linear_trans_z, linear_trans_n, init_state, input_data):
    """
    GRU forward calculation
    args:
        linear_trans_r: linear transformation weights and biases for the R gate
        linear_trans_z: linear transformation weights and biases for the Z gate
        linear_trans_n: linear transformation weights and biases for the candidate hidden state
        init_state: shape [hidden_size], the initial state of the RNN
        input_data: shape [batch_size, time_steps, input_size], input data of `batch_size` sequences, each of
                    which has length `time_steps` and `input_size` features at each time step. 
    returns:
        outputs: shape [batch_size, time_steps, hidden_size], outputs along the sequence. The output at each 
                 time step is exactly the hidden state
        final_state: the final hidden state
    """

    # unpack weights/biases from the three arguments
    (wt_ir, biasir, wt_hr, biashr) = linear_trans_r
    (wt_iz, biasiz, wt_hz, biashz) = linear_trans_z 
    (wt_in, biasin, wt_hn, biashn) = linear_trans_n
    outputs = []
    state = torch.from_numpy(init_state)
    state = torch.squeeze(state)
    inputs = np.transpose(input_data,(1,0,2))
    inputs = torch.from_numpy(inputs)
    batches = input_data.shape[0]
    outputs = np.empty((batches, 0)).tolist()
    i = 0
    

 
    #loop over time steps
    for X in inputs:
      z = np.matmul(X.numpy(),wt_iz) + biasiz + np.matmul(state.numpy(),wt_hz) + biashz
      z = torch.sigmoid(torch.from_numpy(z))

    #compute R gate
      r = np.matmul(X.numpy(),wt_ir) + biasir + np.matmul(state.numpy(),wt_hr) + biashr
      r = torch.sigmoid(torch.from_numpy(r))

    # compute candiate hidden state using the R gate
      temp = np.matmul(state.numpy(),wt_hn) + biashn
      n = torch.from_numpy(np.matmul(X.numpy(),wt_in)) + biasin + torch.mul(r,torch.from_numpy(temp))
      n = torch.tanh(n)
    # compute the final output
      state = torch.mul((1-z),n) + torch.mul(z,state)
      if i == 0:
        i+=1
        for index, s in enumerate(state):
          s = torch.unsqueeze(s,0)                   
          outputs[index] = s
      else:
        for index, s in enumerate(state):
          n = outputs[index]
          s = torch.unsqueeze(s,0)
          m = torch.cat([n,s])
          outputs[index] = m
    outputs = torch.stack(outputs)
    outputs = outputs.numpy()
    state = state.numpy()

        
    
    return outputs, state



def init_gru_with_rnn(wt_h, wt_x, bias):
    """
    This function compute parameters of a GRU such that it performs like a conventional RNN.

    args:
        wt_h: shape [hidden_size, hidden_size], weight matrix for hidden state transformation. Rows corresponds 
              to dimensions of previous hidden states
        wt_x: shape [input_size, hidden_size], weight matrix for input transformation
        bias: shape [hidden_size], bias term

    returns:
        linear_trans_r: linear transformation weights and biases for the R gate
        linear_trans_z: linear transformation weights and biases for the Z gate
        linear_trans_n: linear transformation weights and biases for the candidate hidden state
    """

    #Set the linear transformation for the R gate

    wt_ir  = np.ones_like(wt_x) 
    wt_hr  = np.ones_like(wt_h) 
    biasir = np.zeros_like(bias)
    biashr = np.zeros_like(bias)
    biasir.fill(20000)
    biashr.fill(20000)

    #Set the linear transformation for the Z gate

    wt_iz  = np.ones_like(wt_x) 
    wt_hz  = np.ones_like(wt_h)
    biasiz = np.zeros_like(bias)
    biashz = np.zeros_like(bias)
    biasiz.fill(-20000)
    biashz.fill(-20000)


    #Set the linear transformation for the candidate hidden state

    wt_in  = wt_x
    wt_hn  = wt_h 
    biasin = bias 
    biashn = np.zeros_like(bias)
     
    linear_trans_r = (wt_ir, biasir, wt_hr, biashr)
    linear_trans_z = (wt_iz, biasiz, wt_hz, biashz)
    linear_trans_n = (wt_in, biasin, wt_hn, biashn)

    return linear_trans_r, linear_trans_z, linear_trans_n


def init_gru_with_long_term_memory(input_size, hidden_size):
    """
    This function compute parameters of a GRU such that it maintains the initial state in the memory.  

    args:
        input_size: int, the input dimension 
        hidden_size: int, the hidden dimension

    returns:
        linear_trans_r: linear transformation weights and biases for the R gate
        linear_trans_z: linear transformation weights and biases for the Z gate
        linear_trans_n: linear transformation weights and biases for the candidate hidden state
    """
    #ht-1
    #zt = 1
    wt_x = np.zeros([input_size,hidden_size])
    wt_h = np.zeros([hidden_size,hidden_size])
    bias = np.zeros([hidden_size])
    #Set the linear transformation for the R gate

    wt_ir  = np.ones_like(wt_x) 
    wt_hr  = np.ones_like(wt_h) 
    biasir = np.zeros_like(bias)
    biashr = np.zeros_like(bias)
    biasir.fill(-20000)
    biashr.fill(-20000)

    # Set the linear transformation for the Z gate

    wt_iz  = np.ones_like(wt_x) 
    wt_hz  = np.ones_like(wt_h)
    biasiz = np.zeros_like(bias)
    biashz = np.zeros_like(bias)
    biasiz.fill(20000)
    biashz.fill(20000) 


    #Set the linear transformation for the candidate hidden state 

    wt_in  = np.ones_like(wt_x) 
    wt_hn  = np.ones_like(wt_h) 
    biasin = np.zeros_like(bias) 
    biashn = np.zeros_like(bias) 
     
    linear_trans_r = (wt_ir, biasir, wt_hr, biashr)
    linear_trans_z = (wt_iz, biasiz, wt_hz, biashz)
    linear_trans_n = (wt_in, biasin, wt_hn, biashn)

    return linear_trans_r, linear_trans_z, linear_trans_n


def mha(Wq, Wk, Wv, Wo, input_data):
    """

    args:
        Wq: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head] -> n head total input x edim
        #batch seq input dim
        Wk: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head]
        Wv: a list of matrices, each of which has shape [embed_dim, embed_dim // num_head]
        Wo: a numpy tensor with shape [embed_dim, embed_dim]
        input_data: a tensor with shape [batch_size, sequence_length, embed_dim]. Note that we have the `batch_first` flag on, so the first dimension corresponding to 
                    the batch dimension

    returns:

        output: a tensor with shape [batch_size, sequence_length, embed_dim]

    """

    #using the three matrices to transform X into Q, K V
    Q = input_data
    K = input_data
    V = input_data
    Wq = torch.from_numpy(np.asarray(Wq))
    Wk = torch.from_numpy(np.asarray(Wk))
    Wv = torch.from_numpy(np.asarray(Wv))
    Wo = torch.from_numpy(np.asarray(Wo))
    H = []
    for i in range(Wq.shape[0]):
      Qi = torch.matmul(Q, Wq[i])
      Ki = torch.matmul(K, Wk[i])
      Vi = torch.matmul(V, Wv[i])
      mult = torch.matmul(Qi,torch.transpose(Ki,2,1))
      square = math.sqrt(Wq.shape[2])
      attention_i = torch.softmax(mult / square, dim=2)
      H_i = torch.matmul(attention_i, Vi)
      H.append(H_i)



    H = torch.cat(H,dim=2)
    H = torch.matmul(H,Wo)
   
        
    return H.numpy()
        
