"""A class for an REN cell"""
import numpy as np
import theano
import theano.tensor as T


def intitialise_weights(inputs, outputs, scale=0.1):
    return theano.shared(scale*np.random.randn(inputs, outputs))


class RenCell:

    def __init__(self, emb_dim, sen_len, num_slots, activation=T.nnet.tanh):
        """Initialise all the paramters as shared variables"""
        self.num_slots = num_slots
        self.activation = activation
        self.emb_dim = emb_dim
        self.sen_len = sen_len
        self.F = self._initialize_weights(emb_dim, sen_len, name='mask')
        self.h = self._initialize_weights(num_slots*emb_dim, 1, name='hiddens')
        self.w = self._initialize_weights(num_slots*emb_dim, 1, name='keys')
        self.U = self._initialize_weights(emb_dim, emb_dim, name='U')
        self.V = self._initialize_weights(emb_dim, emb_dim, name='V')
        self.W = self._initialize_weights(emb_dim, emb_dim, name='W')

    def _intitialise_weights(inputs, outputs, name=None, scale=0.1):
        return theano.shared(scale*np.random.randn(inputs, outputs), name=name)

    def _slice(self, item, slot):
        return item[slot*self.emb_dim:slot*self.emb_dim + self.emb_dim, :]

    def _normalize(vector):
        return vector/T.norm(vector)

    def __call__(self, inputs, state):

        def REN_step(self, input, state):
            """ Take mini-bath of inputs and return final sate of the REN Cell
            Inputs - (N,emb_dim,seq_len) matrix
            """
            S = self.F*inputs  # TODO make sure broadcasting is done correctly here
            for slot in range(self.num_slots):
                h_j = self._slice(self.h, slot)
                w_j = self._slice(self.h, slot)
                gate = T.nnet.sigmoid(T.dot(S, h_j) + T.dot(S, w_j))  # should be (N,1)
                _h_j = self.activation(T.dot(S, self.U) + T.dot(h_j, self.V) + T.dot(w_j, self.V) )  # should be (N,emb_dim)
                h_j = h_j + gate*_h_j # need to get the broadcasting right here
                h_j = self._normalize(h_j)



        return output, new_state





def build_simple_rnn(input_dim, hidden_dim, output_dim):

    X = T.dtensor3('X')  # tensor of shape [timesteps, num_data, data_dim]
    h_init = theano.shared(np.random.randn(hidden_dim, 1), name='hidden')  # tensor of shape [hidden_dim]
    N = T.shape(X)[1]

    W1 = intitialise_weights(input_dim, hidden_dim)
    W2 = intitialise_weights(hidden_dim, hidden_dim)
    b1 = theano.shared(0.1*np.random.randn(hidden_dim))
    W3 = intitialise_weights(hidden_dim, output_dim)
    b2 = theano.shared(0.1*np.random.randn(output_dim))

    def rnn_step(x_t, h_tm1, W1, W2, b1, W3, b2):
        h_t = T.nnet.relu(T.dot(x_t, W1) + T.dot(h_tm1, W2) + b1.dimshuffle('x', 0))
        y_t = T.nnet.relu(T.dot(h_t, W3) + b2.dimshuffle('x', 0))
        return h_t, y_t

    [h_vals, Y_vals], updates = theano.scan(rnn_step,
                                             sequences=X,
                                             non_sequences=[W1, W2, b1, W3, b2],
                                             outputs_info=[T.outer(T.ones((N,1)), h_init), None]
                                             )

    rnn_func = theano.function(inputs=[X], outputs=Y_vals)

    return rnn_func

if __name__ == "__main__":
    rnn_func = build_simple_rnn(50, 100, 10)

    print(rnn_func(np.random.randn(1, 32, 50)))
