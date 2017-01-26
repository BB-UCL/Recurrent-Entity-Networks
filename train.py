import numpy as np
from REN import Model
import cPickle
import pdb

params = {'embeding_dimension': 100,
          'num_slots': 20,
          'init_learning_rate': 0.01,
          'num_epochs': 10,
          'vocab_size': 160,
          'batch_size': 33}


def train(path_to_data, path_to_save, params):

    data = np.load(path_to_data)  # npz file obj
    params['max_sent_len'] = max(np.shape(data['stories'])[2],
                                 np.shape(data['queries'])[2])

    Ent_Net = Model.EntityNetwork(params['embeding_dimension'],
                                  params['vocab_size'],
                                  params['num_slots'],
                                  params['max_sent_len'])

    loss = 0.0
    for i in range(params['num_epochs']):
        print(' EPOCH {}').format(i)
        for n,batch in enumerate(get_batch(data, params['batch_size'])):
            batch_loss, accuracy = Ent_Net.train_batch(*batch)
            loss = loss + (batch_loss - loss)/(params['batch_size']*1.0)
            if n%10 ==0:
                print('Loss: {}  training_accuracy: {}').format(batch_loss, accuracy)

    cPickle.dump(Ent_Net, 'Results/model1.save')

def get_batch(data, batch_size):
    stories, queries, _ , answers = data['stories'], data['queries'], data['indices'], data['answers']
    N = np.shape(stories)[0]
    all_indices = np.random.permutation(N)
    for i in range(N/batch_size):
        batch_ind = all_indices[i:i + batch_size]
        yield stories[batch_ind], queries[batch_ind], [1, 3, 5, 7, 9], answers[batch_ind]


if __name__ == "__main__":
    train('Data/train/qa1_single-supporting-fact_train.npz',
          'Results/model1.save', params)
