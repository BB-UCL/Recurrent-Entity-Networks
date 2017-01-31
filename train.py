import numpy as np
from REN import Model
from six.moves import cPickle
import pdb

params = {'embeding_dimension': 100,
          'num_slots': 20,
          'init_learning_rate': 0.01,
          'num_epochs': 20,
          'vocab_size': 160,
          'batch_size': 31}


def train(path_to_train, path_to_test, params):

    train_data = np.load(path_to_train)  # npz file obj
    test_data = extract_stories(np.load(path_to_test))


    params['max_sent_len'] = max(np.shape(train_data['stories'])[2],
                                 np.shape(train_data['queries'])[2])

    Ent_Net = Model.EntityNetwork(params['embeding_dimension'],
                                  params['vocab_size'],
                                  params['num_slots'],
                                  params['max_sent_len'])

    loss = 0.0
    for i in range(params['num_epochs']):
        print(' EPOCH {}').format(i)
        for n,batch in enumerate(get_batch(train_data, params['batch_size'])):
            batch_loss, accuracy = Ent_Net.train_batch(*batch)
            loss = loss + (batch_loss - loss)/(params['batch_size']*1.0)
            if n%10 ==0:
                test_loss, test_accuracy = Ent_Net.test_network(*test_data)
                print('Loss: {}  training_accuracy: {}  test_loss: {} testing_accuracy:{} ').format(batch_loss, accuracy,
                                                                                                    test_loss, test_accuracy)


    #f = open('Results/model1.save', 'wb')
    #cPickle.dump(Ent_Net.params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    #f.close()

def get_batch(data, batch_size):
    stories, queries, indices , answers = extract_stories(data)
    N = np.shape(stories)[0]
    all_indices = np.random.permutation(N)
    for i in range(N/batch_size):
        batch_ind = all_indices[i:i + batch_size]
        yield stories[batch_ind], queries[batch_ind], indices[batch_ind] , answers[batch_ind]

def extract_stories(data):
    return data['stories'], data['queries'], data['indices'], data['answers']



if __name__ == "__main__":
    train('Data/Train/qa7_counting_train.npz',
          'Data/Test/qa7_counting_test.npz', params)