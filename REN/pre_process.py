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


def tokenize(sentence):
    return [token.strip().lower() for token in re.split(r'\s|(?=\.)|(?=\?)',
            sentence, flags=re.VERSION1)]


def parse_stories(story_lines):
    """ Take in the Babi tasks and return a list of
    (story, question_positions, queries, answer) tuples"""

    parsed_stories = []
    story = []
    queries = []
    query_indices = []
    answers = []
    for line in story_lines:
        ID, sentence = line.split(' ', 1)
        ID = int(ID)
        sentence = sentence.strip()
        if ID == 1:
            parsed_stories.append((story, queries, query_indices, answers))
            story = []
            queries = []
            query_indices = []
            answers = []
        if '\t' not in sentence:
            sentence = tokenize(sentence)
            story.append(sentence)
        else:
            query, answer, _ = sentence.split('\t')
            queries.append(tokenize(query))
            query_indices.append(ID-1)
            answers.append(answer)
        if line == story_lines[-1]:
            parsed_stories.append((story, queries, answers))
    return parsed_stories[1:]

def Convert_stories_toKey(stories, vocab_dict):
    """Takes the training stories as a list of tuples and replaces the tokens
    with an integer index in the range of the vocabulary """

    integer_stories = []
    for story, query, answer in stories:
        story = [[vocab_dict[token] for token in sent] for sent in story]
        query = [[vocab_dict[token] for token in sent] for sent in query]
        answer = [vocab_dict[token] for token in answer]
        integer_stories.append((story, query, answer))

    return integer_stories

def get_vocab(stories):
    """
    Recover unique tokens as a vocab and map the tokens to ids.
    """
    tokens_all = []
    for story, query, answer in stories:
        tokens_all.extend([token for sentence in story for token in sentence]
                          + query + [answer])

        vocab = set(tokens_all)
        vocab_dict = {token: i for i, token in enumerate(vocab)}
    return vocab_dict

def pad_stories(stories):
    pass




if __name__ == "__main__":
    with open('../Data/task_1.txt') as task_1:
        stories = task_1.read()
