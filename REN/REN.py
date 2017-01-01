"""A class for an REN cell"""
import numpy as np
import theano
import theano.tensor as T


def intitialise_weights(inputs, outputs, scale=0.1):
    return theano.shared(scale*np.random.randn(inputs, outputs))


class RenCell:

    def __init__(self, emb_dim, num_slots, activation=T.nnet.tanh):
        """Initialise all the paramters as shared variables"""
        self.num_slots = num_slots
        self.activation = activation
        self.emb_dim = emb_dim
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

        def REN_step(S_t, h_tm1, keys, U, V, W):
            """ Take mini-bath of inputs and return final sate of the REN Cell
            Inputs - (N,emb_dim,seq_len) matrix
            """

            h_t = T.zeros_like(h_tm1)
            for slot in range(self.num_slots):
                h_j = self._slice(h_tm1, slot)
                w_j = self._slice(keys, slot)
                gate = T.nnet.sigmoid(T.dot(S, h_j) + T.dot(S, w_j))  # should be (N,1)
                _h_j = self.activation(T.dot(S_t, self.U) + T.dot(h_j, self.V) + T.dot(w_j, self.V) )  # should be (N,emb_dim)
                h_j = h_j + gate*_h_j # need to get the broadcasting right here
                h_j = self._normalize(h_j)
                h_t[slot*self.emb_dim:slot*self.emb_dim + self.emb_dim, :] = h_j

            return h_t, h_t

        [h_vals, Y_vals], updates = theano.scan(REN_step,
                                                 sequences=inputs,
                                                 non_sequences=[self.w, self.U, self.V, self.W],
                                                 outputs_info=[T.outer(T.ones((N,1)), h_init), None]
                                                 )



        return h_vals[-1]
