import numpy as np
from REN import Model
from six.moves import cPickle
import sys
sys.setrecursionlimit(1500)


params = {'embeding_dimension': 100,
          'num_slots': 20,
          'init_learning_rate': 0.01,
          'num_epochs': 1,
          'vocab_size': 160,
          'batch_size': 32}


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
        with open('Results/' + path_to_test[12:-4] + '.txt', 'a+') as f:
            for n, batch in enumerate(get_batch(train_data, params['batch_size'])):
                batch_loss, accuracy = Ent_Net.train_batch(*batch)
                # Get the moving average of the loss
                loss = loss + (batch_loss - loss)/(params['batch_size']*1.0)
                if n % 10 == 0:
                    test_loss, test_accuracy = Ent_Net.test_network(*test_data)
                    f.write('{}   {}  {}  {} \n'.format(batch_loss, accuracy,
                                                        test_loss, test_accuracy))

    with open('Results/model' + path_to_test[12:-4] + '.save', 'wb') as f:
        cPickle.dump(Ent_Net, f, protocol=cPickle.HIGHEST_PROTOCOL)


def get_batch(data, batch_size):
    stories, queries, indices, answers = extract_stories(data)
    N = np.shape(stories)[0]
    all_indices = np.random.permutation(N)
    for i in range(N/batch_size):
        batch_ind = all_indices[i:i + batch_size]
        yield stories[batch_ind], queries[batch_ind], indices[batch_ind], answers[batch_ind]


def extract_stories(data):
    return data['stories'], data['queries'], data['indices'], data['answers']


if __name__ == "__main__":
    data_sets = ['qa1_single-supporting-fact',
                 'qa2_two-supporting-facts',
                 'qa3_three-supporting-facts',
                 'qa4_two-arg-relations',
                 'qa5_three-arg-relations',
                 'qa6_yes-no-questions',
                 'qa7_counting',
                 'qa8_lists-sets',
                 'qa9_simple-negation',
                 'qa10_indefinite-knowledge',
                 'qa11_basic-coreference',
                 'qa12_conjunction',
                 'qa13_compound-coreference',
                 'qa14_time-reasoning',
                 'qa15_basic-deduction',
                 'qa16_basic-induction',
                 'qa17_positional-reasoning',
                 'qa18_size-reasoning',
                 'qa19_path-finding']

    for data_set in data_sets:
        train('Data/Train/' + data_set + '_train.npz',
              'Data/Test/' + data_set + '_test.npz', params)
