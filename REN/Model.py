from REN_Cell import RenCell
import theano
import theano.tensor as T
import numpy as np


class EntityNetwork():

    def __init__(self, emb_dim, vocab_size, num_slots):
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.num_slots = num_slots  # num of hdn state units.
        self.cell = RenCell(self.emb_dim, self.num_slots)

        # Paceholders for input
        self.Stories = T.itensor3(name='Stories')  # Num_stories x T x K
        self.Queries = T.imatrix3(name='Queries')  # Num_stories X K_max
        self.Answers = T.imatrix3(name='Answers')  # Num_stories X K_max

        # Data Set dimensions
        self.N = T.shape(self.Stories)[0]
        self.T = T.shape(self.Stories)[1]  # max_story_len
        self.K = T.shape(self.Stories)[2]  # max_sent_len

        # Build the Computation Graph
        self._create_network()

    def _initialize_weights(self, *args, **kwargs):
        pass

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

    def _create_network(self):

        # Embed the stories and queries
        self.Emb_Matrix = self._initialize_weights(self.vocab_size, self.emb_dim)
        self.Emb_Stories = self.Emb_Matrix[self.Stories]  # Shared variable of shape (N_stories, T_max, K_max,emb_dim)
        self.Emb_Queries = self.Emb_Matrix[self.Queries]  # shape (N,K,emb_dim)

        # Initialise mask and mask stories
        self.F = self._initialize_weights(self.K, self.emb_dim)
        self.masked_stories = T.sum((self.Emb_Stories*self.F.dimshuffle(['x', 'x', 0, 1])),
                                    axis=2)  # shape (N_stories, T_max, emb_dim

        self.masked_queries = T.sum((self.Emb_Queries*self.F.dumshuffle(['x', 0, 1])),
                                    axis=1)  # shape N,emb_dim

        # Initialise state of the entity network
        init_state = self._initialize_weights(self.N, self.num_slots,
                                              self.emb_dim, name="hidden_state")

        init_keys = self._initialize_weights(self.N, self.num_slots,
                                             self.emb_dim, name="init_keys")

        # Pass stories through recurrent entity Cell
        self.h_T, _ = self.cell(self.masked_stories, init_state, init_keys)

        # Use the ouput to generate an answer
        self.hop_weight = self._initialize_weights(self.emb_dim, self.emb_dim)
        self.out_weight = self._initialize_weights(self.emb_dim, self.vocab_size)

        self._answer = self._get_answer(self.h_T, self.Emb_Queries,
                                        self.hop_weight, self.out_weight)

        # Define the loss function
        self.loss = T.nnet.categorical_crossentropy(self._answer, self.Answers)

    def train_batch(self, Stories, Queries, Answers):
        pass
