from nose.tools import *
import numpy as np
import theano
import theano.tensor as T
import REN.REN_Cell as Cell

def setup():
    print "SETUP!"

def teardown():
    print "TEAR DOWN!"

def test_basic():
    print "I RAN!"

def test_REN_Cell():
    cell = Cell.RenCell(31, 10)  # emb_dim, num_slots
    _input = np.random.randn(11, 100, 31)  # T,N,D
    x = T.dtensor3('x')
    H_init = theano.shared(np.zeros((100, 10, 31)), name='Initial Hidden')
    keys_init = theano.shared(np.zeros((100, 10, 31)), name='Initial keys')
    y, _ = cell(x, H_init, keys_init)
    f = theano.function(inputs=[x], outputs=[y])
    f(_input)
