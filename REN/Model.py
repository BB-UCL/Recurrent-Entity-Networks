import REN_Cell
import theano
import theano.tensor as T

def mask(masks, input):
    return T.sum(inputs*mask.dimshuffle('x', 'x', 0, 1), axis=3)

# Load Data
Emb_Sent = load_data(path_to_data)  # should be of shape (T,N,D,K)

# Mask Sentences
inputs = T.ftensor4(name='inputs')
queries = T.ftensor4(name='queries')

mask = theano.shared(np.random.randn(emb_dim, sent_len))
masked_input = mask(mask, inputs)  # need to specify these values
masked_queries = mask(mask, queries)  # should be shape (T, N, emb_dim)

# Pass through the Reccurrent Entity Cell
REN = REN_Cell.RenCell(emb_dim, num_slots)
ouputs = REN(masked_input)  # should be of shape (N, emb_dim*num_slots)

# Create loss function
