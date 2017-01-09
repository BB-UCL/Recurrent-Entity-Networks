"""A class for an REN cell"""
import numpy as np
import theano
import theano.tensor as T
import util as u


class RenCell:
    """ The Core Recurrent Entity Network Cell.
    """


    def __init__(self, emb_dim, num_slots, activation=T.nnet.sigmoid):
        """Initialise all the paramters as shared variables"""
        self.num_slots = num_slots  # M
        self.activation = activation
        self.emb_dim = emb_dim  # J

        # Initialise Parameters
        self.U = self._initialize_weights(emb_dim, emb_dim, name='U')
        self.V = self._initialize_weights(emb_dim, emb_dim, name='V')
        self.W = self._initialize_weights(emb_dim, emb_dim, name='W')

    def _initialize_weights(self, inputs, outputs, name=None, scale=0.1):
        return theano.shared(scale*np.random.randn(inputs, outputs), name=name)

    def _get_gate(self, S_t, H, Keys):
        S_t = S_t.dimshuffle([0, 'x', 1])
        return T.nnet.sigmoid(T.sum(H*S_t + Keys*S_t, axis=2, keepdims=True))

    def _get_candidate(self, S_t, H, Keys):
        return self.activation(T.dot(S_t, self.U).dimshuffle([0, 'x', 1]) +
                               T.dot(H, self.V) + T.dot(Keys, self.W))

    def _update_memory(self, H, _H, gate):
        _H_prime = H + gate*_H
        return _H_prime/(_H_prime.norm(2, axis=2).dimshuffle([0, 1, 'x']))

    def __call__(self, inputs, init_state, init_keys):
        """ Take mini-bath of inputs and return final sate of the REN Cell
        Inputs - (Time_steps, N, emb_dim) matrix
        """

        N = T.shape(inputs)[0]
        assert(T.shape(init_state) == (N, self.num_slots, self.emb_dim),
               """The dimensions of the hidden state needs to be (batch_size,
                  num_slots, embd_dim)""")

        def REN_step(S_t, H_tm1, Keys, U, V, W):
            """ Perfrom one step of the RNN updates"""

            gate = self._get_gate(S_t, H_tm1, Keys)  # should be (N,emb_dim)

            _H = self._get_candidate(S_t, H_tm1, Keys)  # should be (N,emb_dim)

            H = self._update_memory(H_tm1, _H, gate)

            return H

        Y_vals, updates = theano.scan(REN_step,
                                      sequences=inputs,
                                      outputs_info=[init_state],
                                      non_sequences=[init_keys, self.U, self.V, self.W],
                                      )

        return Y_vals[-1], updates
