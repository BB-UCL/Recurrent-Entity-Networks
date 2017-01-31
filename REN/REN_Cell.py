"""A class for an REN cell"""
import numpy as np
import theano
import theano.tensor as T
import util as u


class RenCell:
    """ The Core Recurrent Entity Network Cell. As described in
    arXiv:1612.03969v1.
    """

    def __init__(self, emb_dim, num_slots, activation=T.nnet.relu):
        """Initialise all the paramters as shared variables"""
        self.num_slots = num_slots  # M
        self.activation = activation
        self.emb_dim = emb_dim  # J

        # Initialise Parameters
        self.U = self._initialize_weights(emb_dim, emb_dim, name='U')
        self.V = self._initialize_weights(emb_dim, emb_dim, name='V')
        self.W = self._initialize_weights(emb_dim, emb_dim, name='W')
        self.a = theano.shared(1.0)  # Prelu gradient
        self.params = {'U': self.U, 'V': self.V, 'W': self.W, 'a': self.a}

    def _initialize_weights(self, inputs, outputs, name=None, scale=0.1):
        return theano.shared(scale*np.random.randn(inputs, outputs), name=name)

    def _get_gate(self, S_t, H, Keys):
        """ Equation (2) in arXiv:1612.03969v1"""
        S_t = S_t.dimshuffle([0, 'x', 1])
        return T.nnet.sigmoid(T.sum(H*S_t + Keys*S_t, axis=2, keepdims=True))

    def _get_candidate(self, S_t, H, Keys):
        """ Equation (3) in arXiv:1612.03969v1"""
        return self.activation(T.dot(S_t, self.U).dimshuffle([0, 'x', 1]) +
                               T.dot(H, self.V) + T.dot(Keys, self.W))

    def _update_memory(self, H, _H, gate):
        """ Equation (4)/(5) in arXiv:1612.03969v1"""
        _H_prime = H + gate*_H
        return _H_prime/(_H_prime.norm(2, axis=2).dimshuffle([0, 1, 'x']))

    def __call__(self, inputs, init_state, init_keys, indices=-1):
        """ Take mini-bath of inputs and return the sate of the REN Cell
        at the time-steps specified by indices.
            inputs - (Time_steps, N_stories, emb_dim) tensor
            init_state - (Time_steps, N_stories, num_slots, emb_dim) tensor
            init_keys - (Time_steps, N_stories, num_slots, emb_dim) tensor
            indices - (N_stories, N_questions)

            output - (N_stories, N_questions, num_slots, emb_dim) tensor
        """
        story_indices = T.arange(T.shape(inputs)[1]).dimshuffle([0, 'x'])

        def REN_step(S_t, H_tm1, Keys, U, V, W):
            """ Perfrom one step of the RNN updates"""

            gate = self._get_gate(S_t, H_tm1, Keys)  # should be (N, M, emb_dim)

            _H = self._get_candidate(S_t, H_tm1, Keys)  # should be (N, M, emb_dim)

            H = self._update_memory(H_tm1, _H, gate)

            return H

        out_vals, updates = theano.scan(REN_step,
                                        sequences=inputs,
                                        outputs_info=[init_state],
                                        non_sequences=[init_keys,
                                                       self.U,
                                                       self.V,
                                                       self.W],
                                        )

        return out_vals[indices, story_indices], updates
