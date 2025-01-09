import re, collections

# from Neural Machine Translation of Rare Words with Subword Units

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2, 'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 10


# first merge
# pairs:
# defaultdict(<class 'int'>, {('l', 'o'): 7, ('o', 'w'): 7, ('w', '</w>'): 5, ('w', 'e'): 8,
#                             ('e', 'r'): 2, ('r', '</w>'): 2, ('n', 'e'): 6, ('e', 'w'): 6,
#                             ('e', 's'): 9, ('s', 't'): 9, ('t', '</w>'): 9, ('w', 'i'): 3,
#                             ('i', 'd'): 3, ('d', 'e'): 3})
#
# best: ('e', 's')
#
# vocab: {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}


#     vocab: {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
#     merge 0: best = ('e', 's')
#
#     vocab: {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
#     merge 1: best = ('es', 't')
#
#     vocab: {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est </w>': 6, 'w i d est </w>': 3}
#     merge 2: best = ('est', '</w>')
#
#     vocab: {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
#     merge 3: best = ('l', 'o')
#
#     vocab: {'lo w </w>': 5, 'lo w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
#     merge 4: best = ('lo', 'w')
#
#     vocab: {'low </w>': 5, 'low e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
#     merge 5: best = ('n', 'e')
#
#     vocab: {'low </w>': 5, 'low e r </w>': 2, 'ne w est</w>': 6, 'w i d est</w>': 3}
#     merge 6: best = ('ne', 'w')
#
#     vocab: {'low </w>': 5, 'low e r </w>': 2, 'new est</w>': 6, 'w i d est</w>': 3}
#     merge 7: best = ('new', 'est</w>')
#
#     vocab: {'low </w>': 5, 'low e r </w>': 2, 'newest</w>': 6, 'w i d est</w>': 3}
#     merge 8: best = ('low', '</w>')
#
#     vocab: {'low</w>': 5, 'low e r </w>': 2, 'newest</w>': 6, 'w i d est</w>': 3}
#     merge 9: best = ('w', 'i')

for i in range(num_merges):
    # count number of occurences of each pair of letters
    pairs = get_stats(vocab)

    best = max(pairs, key=pairs.get)

    print("\nvocab: {}".format(vocab))

    vocab = merge_vocab(best, vocab)

    print("merge {}: best = {}". format(i, best))
