from REN_Cell import RenCell
import theano
import theano.tensor as T
import numpy as np
import lasagne


class EntityNetwork():

    def __init__(self, emb_dim, vocab_size, num_slots, max_sent_len, optimiser=lasagne.updates.adam):
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_slots = num_slots  # num of hdn state units.
        self.cell = RenCell(self.emb_dim, self.num_slots)
        self.optimiser = optimiser

        # Paceholders for input
        self.Stories = T.ltensor3(name='Stories')  # Num_stories x T x K_max
        self.Queries = T.ltensor3(name='Queries')  # Num_stories X Num_queries X K_max
        self.Indices = T.lmatrix(name="Indices")  # Num_Stories X Num_queries
        self.Answers = T.lmatrix(name='Answers')  # Num_stories X Num_queries

        # Data Set dimensions
        self.N = T.shape(self.Stories)[0]
        self.K = max_sent_len

        # Combine cell parameters with all the other parameters and init
        self.params = self.cell.params
        self.params.update(self._initialise_weights())

        # Build the Computation Graph and get the training function
        self._create_network(self.params)
        self.train_func = self._get_train_func()
        self.answer_func = self._get_answer_func()

    def _create_network(self, params):

        # Embed the stories and queries
        self.Emb_Stories = params['emb_matrix'][self.Stories]  # Shared variable of shape (N_stories, T_max, K_max, emb_dim)
        self.Emb_Queries = params['emb_matrix'][self.Queries]  # shape (N, Num_q, K,emb_dim)

        # mask stories and queries
        self.masked_stories = T.sum((self.Emb_Stories*params['mask'].dimshuffle(['x', 'x', 0, 1])),
                                    axis=2).dimshuffle([1, 0, 2])  # shape (T_max, N_stories, emb_dim)

        self.masked_queries = T.sum((self.Emb_Queries*params['mask'].dimshuffle(['x', 'x', 0, 1])),
                                    axis=2)  # shape N,emb_dim

        # Initialise Hidden state
        init = self._init_weight([self.num_slots, self.emb_dim],
                                 name="init_keys").dimshuffle(['x', 0, 1])

        one_repeats = T.ones([self.N]).dimshuffle([0, 'x', 'x'])
        self.params['init_keys'] = init*one_repeats  # shape (N_stories, num_slots, emb_dim)
        self.init_state = init*one_repeats

        # Pass through the recurrent entity cell
        self.H_s, _ = self.cell(self.masked_stories,
                                self.init_state,
                                self.init_keys,
                                self.Indices)

        # Reshape H to have dimension (N_q*N_s, M, D) and Q to be N_q*N_s, D
        #self.H_s = self.H_s.dimshuffle([1, 0, 2, 3])
        self.H_s = T.reshape(self.H_s, [-1, self.num_slots, self.emb_dim], ndim=3)
        self.masked_queries = T.reshape(self.masked_queries,
                                        [-1, self.emb_dim],
                                        ndim=2)  # N_q*N_s, D

        # Use the ouput to generate a set of answers
        self._answers = self._get_answer(self.H_s, self.masked_queries,
                                         params['hop_weight'],
                                         params['out_weight'])  # (N*N_q, vocab_size)

        # Define the loss function and get the accuracy
        self.loss = T.mean(T.nnet.categorical_crossentropy(T.nnet.softmax(self._answers),
                                                          self.Answers.flatten()))

        self.accuracy = T.mean(T.eq(T.argmax(self._answers, axis=1), self.Answers.flatten()))

    def _initialise_weights(self):
        params = {}
        params['emb_matrix'] = self._init_weight([self.vocab_size, self.emb_dim])
        params['mask'] = theano.shared(np.ones([self.K, self.emb_dim]))
        params['hop_weight'] = self._init_weight([self.emb_dim, self.emb_dim])
        params['out_weight'] = self._init_weight([self.emb_dim, self.vocab_size])
        return params

    def _init_weight(self, shape, name=None, scale=0.1):
        return theano.shared(scale*np.random.normal(size=shape), name=name)

    def _get_answer(self, h_T, queries, hop_wt, out_wt):

        # Attend over last state with embedded query vector
        attn = T.nnet.softmax(T.sum(queries.dimshuffle([0, 'x', 1])*h_T,
                                    axis=2))  # shape (N * N_q, num_slots)

        # Weight memories by attention
        u = T.sum(attn.dimshuffle([0, 1, 'x'])*h_T, axis=1)  # (N*N_q, emb_dim)

        # Get answer (It might be interesting to replace out_wt with embedding mask)
        answers = T.dot(T.tanh(T.dot(u, hop_wt) + queries), out_wt)  # shape (N*N_q, Vocab_size)

        return answers

    def _get_train_func(self):
        updates = self.optimiser(self.loss, self.params.values())
        return theano.function(inputs=[self.Stories, self.Queries, self.Indices,
                                       self.Answers],
                               outputs=[self.loss, self.accuracy],
                               updates=updates)

    def _get_answer_func(self):
        return theano.function(inputs=[self.Stories, self.Queries,
                                       self.Indices],
                               outputs=self._answers)

    def train_batch(self, Stories, Queries, Indices, Answers):
        loss, accuracy = self.train_func(Stories, Queries, Indices, Answers)
        return loss, accuracy

    def test_network(self, Stories, Queries, Indices, Answers):
        f = theano.function(inputs=[self.Stories, self.Queries, self.Indices,
                                       self.Answers],
                               outputs=[self.loss, self.accuracy],)
        loss, accuracy = f(Stories, Queries, Indices, Answers)
        return loss, accuracy

    def get_answer(self, Stories, Queries):
        return self.answer_func(Stories, Queries)

if __name__ == "__main__":
    ENT = EntityNetwork(10, 160, 15, 10)
