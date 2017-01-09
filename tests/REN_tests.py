from nose.tools import *
import numpy as np
import theano
import theano.tensor as T
import REN
import REN.REN_Cell as Cell

def setup():
    print "SETUP!"

def teardown():
    print "TEAR DOWN!"

def test_basic():
    print "I RAN!"

def test_REN_Cell():
    cell = Cell.RenCell(30, 10)
    _input = np.random.randn(10, 100, 30)
    x = T.dtensor3('x')
    y = cell(x)
    f = theano.function(inputs=[x], outputs=[y])
    f(_input)
