from REN import pre_process


def test_tokenize():
    tokens = pre_process.tokenize('RAza went to the Park?')
    assert(tokens == ['raza', 'went', 'to', 'the', 'park', '?'])

def test_parse_stories():
    """ This test is now out_of date. Rewrite needed."""
    story_lines = ['1 Daniel journeyed to the office.\n',
                   '2 John moved to the bedroom.\n',
                   '3 Where is John?\tbedroom\t2\n',
                   '4 Mary moved to the garden.\n',
                   '5 John moved to the garden.\n',
                   '6 Where is Mary?\tgarden\t4\n',
                   '7 Daniel travelled to the garden.\n',
                   '8 Mary went to the office.\n',
                   '9 Where is John?\tgarden\t5\n',
                   '10 John journeyed to the kitchen.\n',
                   '11 Mary went to the garden.\n',
                   '12 Where is Daniel?\tgarden\t7\n',
                   '13 John travelled to the office.\n',
                   '14 Mary journeyed to the bedroom.\n',
                   '15 Where is Mary?\tbedroom\t14\n']
    parsed = pre_process.parse_stories(story_lines)

    assert(parsed == [([['daniel', 'journeyed', 'to', 'the', 'office', '.'],
                        ['john', 'moved', 'to', 'the', 'bedroom', '.'],
                        ['mary', 'moved', 'to', 'the', 'garden', '.'],
                        ['john', 'moved', 'to', 'the', 'garden', '.'],
                        ['daniel', 'travelled', 'to', 'the', 'garden', '.'],
                        ['mary', 'went', 'to', 'the', 'office', '.'],
                        ['john', 'journeyed', 'to', 'the', 'kitchen', '.'],
                        ['mary', 'went', 'to', 'the', 'garden', '.'],
                        ['john', 'travelled', 'to', 'the', 'office', '.'],
                        ['mary', 'journeyed', 'to', 'the', 'bedroom', '.']],
                       [(2, ['where', 'is', 'john', '?']),
                        (5, ['where', 'is', 'mary', '?']),
                        (8, ['where', 'is', 'john', '?']),
                        (11, ['where', 'is', 'daniel', '?']),
                        (14, ['where', 'is', 'mary', '?'])],
                       ['bedroom', 'garden', 'garden', 'garden', 'bedroom'])])
