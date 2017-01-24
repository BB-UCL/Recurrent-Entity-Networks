""" Pre-Processing Pipe-Line:

1) Tokenize
    - Load Data
    - Split into individual words.
2) Map to integers in the range 1-vocab_size
3) Format into the right shape for input to the REN
    - Pad stories/sentences to right length
    - reshape

Vocab size is 159

output will be a a series of numpy arrays to be fed as input
to the network.
--> Stories Array (Num_Stories, max_story_length, max_sent_lenth)
--> Queries Array (Num_stories, Num_queries, max_sent_length)
--> Indices (Num_stories, Num_queries)
--> Answers Array (Num_stories, Num_queries)
"""
import regex as re
import numpy as np
import copy
PAD_TOKEN = 'PAD'

# TODO group stories currently expects same number of questions per story.


def tokenize(sentence):
    return [token.strip().lower() for token in re.split(r'\s|(?=\.)|(?=\?)',
            sentence, flags=re.VERSION1) if token.strip()]


def parse_stories(story_lines):
    """ Take in the Babi tasks and return a list of
    (story,queries, query_position, answers) tuples as
    well as the length of the longest story and sentence"""
    parsed_stories = []
    story = []
    queries = []
    query_indices = []
    answers = []
    max_story = 0
    max_sent = 0
    query_count = 0
    for n, line in enumerate(story_lines):
        ID, sentence = line.split(' ', 1)
        ID = int(ID)
        sentence = sentence.strip()
        if ID == 1 and n>1:
            parsed_stories.append((story, queries, query_indices, answers))
            if len(story) > max_story:
                max_story = len(story)
            story = []
            queries = []
            query_indices = []
            answers = []
            query_count = 0
        if '\t' not in sentence:
            sentence = tokenize(sentence)
            if len(sentence) > max_sent:
                max_sent = len(sentence)
            story.append(sentence)
        else:
            query, answer, _ = sentence.split('\t')
            query = tokenize(query)
            query_count += 1
            if len(query) > max_sent:
                max_sent = len(query)
            queries.append(query)
            query_indices.append(ID-query_count-1)
            answers.append(answer)
    parsed_stories.append((story, queries, query_indices, answers))

    return parsed_stories, max_story, max_sent


def convert_stories_tokens(stories, vocab_dict):
    """Takes the training stories as a list of tuples and replaces the tokens
    with an integer index in the range of the vocabulary """

    integer_stories = []
    for story, query, index, answer in stories:
        story = [[vocab_dict[token] for token in sent] for sent in story]
        query = [[vocab_dict[token] for token in sent] for sent in query]
        answer = [vocab_dict[token] for token in answer]
        integer_stories.append((story, query, index, answer))

    return integer_stories


def get_vocab_dict(parsed_stories, vocab_dict=None):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    if vocab_dict is None:
        vocab_dict = {PAD_TOKEN: 0}
    tokens_all = []
    for story, queries, _, answers in parsed_stories:
        tokens_all.extend([token for sentence in story for token in sentence]
                          + [token for query in queries for token in query]
                          + [token for token in answers])
    new_vocab = sorted(set(tokens_all))
    for word in new_vocab:
        if word not in vocab_dict:
            vocab_dict[word] = len(vocab_dict) + 1
    return vocab_dict


def pad_stories(stories, max_sent_len, max_story_len):
    padded_stories = copy.deepcopy(stories)
    for story, queries, index, answer in padded_stories:
        for query in queries:
            query.extend([PAD_TOKEN]*(max_sent_len - len(query)))
        for sentence in story:
            sentence.extend([PAD_TOKEN]*(max_sent_len-len(sentence)))
        story.extend([[PAD_TOKEN]*max_sent_len]*(max_story_len - len(story)))
    return padded_stories


def group(parsed_stories):
    """Take a list of (story, query, indices, answer) tuples of type
    integer_stories and return 4 arrays of N stories, N*num_q queries,
    N*num_q indices and N*num_q answers"""
    groups = []
    for i in range(4):
        group = [np.asarray(story_group[i]) for story_group in parsed_stories]
        group = np.stack(group, axis=0)
        groups.append(group)
    return groups


def save_parsed_data(groups, filename):
    np.savez(filename, stories=groups[0], queries=groups[1], indices=groups[2],
             answers=groups[3])


def main():
    vocab_dict = None
    filenames = ['qa1_single-supporting-fact',
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
                 'qa19_path-finding']

    for filename in filenames:
        with open('Data/en-10k/' + filename + '_train.txt') as f:
            storylines_train = f.readlines()

        # Parse Stories
        print("parsing " + filename)
        parsed_stories, max_story, max_sent = parse_stories(storylines_train)

        # Update Vocabulary
        print("Learning new words")
        vocab_dict = get_vocab_dict(parsed_stories, vocab_dict)

        # Pad Stories
        print("Padding Stories")
        padded_stories = pad_stories(parsed_stories, max_sent, max_story)

        # Convert to ints
        print("Mapping to Integers")
        int_stories = convert_stories_tokens(padded_stories, vocab_dict)

        # Reshape ready for model
        print("Reshaping for Neural Net Input")
        grouped_stories = group(int_stories)

        # Save model
        print('Saving')
        save_parsed_data(grouped_stories, 'Data/Train/' + filename + '_train')

    print("vocab is of size {}".format(len(vocab_dict)))

if __name__ == "__main__":
    main()
