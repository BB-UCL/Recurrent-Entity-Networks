from REN_Cell import RenCell
import theano
import theano.tensor as T
import numpy as np


def model(emb_dim, vocab_size, num_slots):

    # Placeholders for input
    Stories = T.itensor3(name='Stories')  # Num_stories x T_max x K_max
    Queries = T.imatrix3(name='Queries') # Num_stories X K_max
    answers = T.imatrix3(name='Answers')
    N = T.shape(Stories)[0]
    K = T.shape(Stories)[2]  # max_sent_len
    T = T.shape(Stories)[1]  # max_story_len

    # Embed the stories and queries
    Emb_Matrix = theano.shared(0.1*np.random.rand(vocab_size, emb_dim))
    Emb_Stories = Emb_Matrix[Stories]  # Shared variable of shape (N_stories, T_max, K_max,emb_dim)
    Emb_Queries = Emb_Matrix[Queries]  # shape (N,K,emb_dim)

    # Initialise mask
    F = theano.shared(0.1*np.random.rand(K, emb_dim))
    masked_stories = T.sum((Emb_Stories*F.dimshuffle(['x', 'x', 0, 1])), axis=2) # shape (N_stories, T_max, emb_dim)
    masked_queries = T.sum((Emb_Queries*F.dumshuffle(['x', 0, 1])), axis=1)  # shape N,emb_dim

    # Initialise state of the entity network
    init_state = theano.shared(0.1*np.random.rand(N, num_slots, emb_dim), name="hidden_state")
    init_keys = theano.shared(0.1*np.random.rand(N, num_slots, emb_dim), name="init_keys")

    # Pass stories through recurrent entity Cell
    cell = RenCell(emb_dim, num_slots)
    h_T, cell_updates = cell(masked_stories, init_state, init_keys)

    # Use the ouput to generate an answer
    p = T.nnet.softmax(T.sum(Queries.dimshuffle([0, 'x', 1])*h_T, axis=2))  # shape (N,num_slots)
    u = T.sum(p.dimshuffle([0, 1, 'x'])*h_T, axis=1) # shape (N, emb_dim)
    hop_weight = theano.shared(0.1*np.random.rand(emb_dim, emb_dim))
    out_weight = theano.shared(0.1*np.random.rand(emb_dim, vocab_size))
    _answers = T.nnet.softmax(T.dot(T.tanh(T.dot(u, hop_weight) + Queries), out_weight)) #  shape (N, Vocab_size)

    # Define the loss function
    loss = T.nnet.categorical_crossentropy(_answers, answers)


    return None
