from REN_Cell import RenCell
import theano
import theano.tensor as T
import numpy as np
import lasagne


class EntityNetwork():

    def __init__(self, emb_dim, vocab_size, num_slots, optimiser=lasagne.updates.adam):
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_slots = num_slots  # num of hdn state units.
        self.cell = RenCell(self.emb_dim, self.num_slots)
        self.optimser = optimiser
        self.params = [self.cell.params, self._initialise_weights()]

        # Paceholders for input
        self.Stories = T.itensor3(name='Stories')  # Num_stories x T x K
        self.Queries = T.imatrix3(name='Queries')  # Num_stories X K_max
        self.Answers = T.imatrix3(name='Answers')  # Num_stories X K_max

        # Data Set dimensions
        self.N = T.shape(self.Stories)[0]
        self.T = T.shape(self.Stories)[1]  # max_story_len
        self.K = T.shape(self.Stories)[2]  # max_sent_len

        # Build the Computation Graph and get the training function
        self._create_network(self.params)
        self.train_func = self._get_train_func()
        self.answer_func = self._get_answer_func()

    def _initialise_weights(self, *args, **kwargs):
        params = {}
        params['Emb_Matrix'] = self._init_weight([self.vocab_size, self.emb_dim])
        params['mask'] = self._init_weight([self.K, self.emb_dim])

        params['init_keys'] = self._init_weight([self.N, self.num_slots,
                                                self.emb_dim], name="init_keys")
        params['hop_weight'] = self._init_weight([self.emb_dim, self.emb_dim])
        params['out_weight'] = self._init_weight([self.emb_dim, self.vocab_size])

        return params

    def _init_weight(self, shape, name=None, scale=0.1):
        return theano.shared(scale*np.random.normal(size=shape), name=name)

    def _get_answer(self, h_T, queries, hop_wt, out_wt):

        # Attend over last state with embedded query vector
        attn = T.nnet.softmax(T.sum(queries.dimshuffle([0, 'x', 1])*h_T,
                                    axis=2))  # shape (N,num_slots)

        # Weight memories by attention
        u = T.sum(attn.dimshuffle([0, 1, 'x'])*h_T, axis=1)  # (N, emb_dim)

        # Get answer
        answer = T.nnet.softmax(T.dot(T.tanh(T.dot(u, hop_wt) + queries),
                                      out_wt))  # shape (N, Vocab_size)

        return answer

    def _get_train_func(self):
        updates = self.optimser(self.loss, self.params)
        return theano.function(inputs=[self.stories, self.Queries, self.Answers],
                               outputs=[self.loss, self.accuracy], updates=updates)

    def _get_answer_func(self):
        return theano.function(inputs=[self.Stories, self.Queries],
                               outputs=self._answers)

    def _create_network(self, params):

        # Embed the stories and queries
        self.Emb_Stories = params['Emb_Matrix'][self.Stories]  # Shared variable of shape (N_stories, T_max, K_max,emb_dim)
        self.Emb_Queries = params['Emb_Matrix'][self.Queries]  # shape (N,K,emb_dim)

        # Initialise mask and mask stories
        self.masked_stories = T.sum((self.Emb_Stories*params['mask'].dimshuffle(['x', 'x', 0, 1])),
                                    axis=2)  # shape (N_stories, T_max, emb_dim

        self.masked_queries = T.sum((self.Emb_Queries*params['mask'].dumshuffle(['x', 0, 1])),
                                    axis=1)  # shape N,emb_dim

        # Pass stories through recurrent entity Cell
        init_state = self._init_weight([self.N, self.num_slots,
                                        self.emb_dim], name="init_state")

        self.h_T, _ = self.cell(self.masked_stories,
                                init_state, params['init_keys'])

        # Use the ouput to generate an answer
        self._answer = self._get_answer(self.h_T, self.Emb_Queries,
                                        self.hop_weight, self.out_weight)

        # Define the loss function
        self.loss = T.nnet.categorical_crossentropy(self._answer, self.Answers)
        self.accuracy = T.mean(T.eq(T.argmax(self._answer, axis=1), self.Answers))

    def train_batch(self, Stories, Queries, Answers):
        return self.train_func(Stories, Queries, Answers)

    def get_answer(self, Stories, Queries):
        return self.answer_func(Stories, Queries)
