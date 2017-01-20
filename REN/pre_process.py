""" Pre-Processing Pipe-Line:

1) Tokenize
    - Load Data
    - Split into individual words.
2) Map to integers in the range 1-vocab_size
3) Format into the right shape for input to the REN
    - Pad stories/sentences to right length
    - reshape

output will be a a series of numpy arrays to be fed as input
to the network.
--> Stories Array (Num_Stories, max_story_length, max_sent_lenth)
--> Queries Array (Num_storeis, max_sent_length)
--> Answers Array (Num_stories,1)
"""
import regex as re
import numpy as np
import copy
PAD_TOKEN = 'PAD'


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

    for n, line in enumerate(story_lines):
        ID, sentence = line.split(' ', 1)
        ID = int(ID)
        sentence = sentence.strip()
        if ID == 1 and n > 1:
            parsed_stories.append((story, queries, query_indices, answers))
            if len(story) > max_story:
                max_story = len(story)
            story = []
            queries = []
            query_indices = []
            answers = []
        if '\t' not in sentence:
            sentence = tokenize(sentence)
            if len(sentence) > max_sent:
                max_sent = len(sentence)
            story.append(sentence)
        else:
            query, answer, _ = sentence.split('\t')
            query = tokenize(query)
            if len(query) > max_sent:
                max_sent = len(query)
            queries.append(query)
            query_indices.append(ID-1)
            answers.append(answer)
    parsed_stories.append((story, queries, query_indices, answers))

    return parsed_stories, max_story, max_sent


def convert_stories_toints(stories, vocab_dict):
    """Takes the training stories as a list of tuples and replaces the tokens
    with an integer index in the range of the vocabulary """

    integer_stories = []
    for story, query, index, answer in stories:
        story = [[vocab_dict[token] for token in sent] for sent in story]
        query = [[vocab_dict[token] for token in sent] for sent in query]
        answer = [vocab_dict[token] for token in answer]
        integer_stories.append((story, query, index, answer))

    return integer_stories


def get_vocab_dict(parsed_stories):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    tokens_all = []
    for story, queries, _, answers in parsed_stories:
        tokens_all.extend([token for sentence in story for token in sentence]
                          + [token for query in queries for token in query]
                          + [token for token in answers])
    vocab = set(sorted(tokens_all))
    vocab_dict = {token: i for i, token in enumerate(vocab)}
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


if __name__ == "__main__":
    with open('../Data/task_1.txt') as task_1:
        stories = task_1.read()
