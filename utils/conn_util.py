import json
import sys
import util

def get_doc_conns(doc):
    conn_list = [] #[(sent_index, conn_indices), .. ]
    for sent_index, sentence in enumerate(doc["sentences"]):
        sent_words_list = [word[0] for word in sentence["words"]]
        for conn_indices in _check_connectives(sent_words_list): #[[2, 3], [0]]
            conn_list.append((sent_index, conn_indices))
    return conn_list


# identify connectives in sentence (sent_tokens)
# return indices: [[2, 3], [0]]
def _check_connectives(sent_tokens):
    sent_tokens = [word.lower() for word in sent_tokens ]
    indices = []
    tagged = set([])
    sortedConn = util.get_sorted_conns_list()
    for conn in sortedConn:
        if '..' in conn:
            c1, c2 = conn.split('..')
            c1_indice = util.getSpanIndecesInSent(c1.split(), sent_tokens)#[[7]]
            c2_indice = util.getSpanIndecesInSent(c2.split(), sent_tokens)#[[10]]
            if c1_indice!= [] and c2_indice != []:
                if c1_indice[0][0] < c2_indice[0][0]:
                    temp = set([t for t in (c1_indice[0]+c2_indice[0]) ])
                    if tagged & temp == set([]):
                        indices.append(c1_indice[0]+c2_indice[0])# [[7], [10]]
                        tagged = tagged.union(temp)
        else:
            c_indice = util.getSpanIndecesInSent(conn.split(), sent_tokens)#[[2,6],[1,3],...]
            if c_indice !=[]:
                tt = []
                for item in c_indice:
                    if set(item) & tagged == set([]):
                        tt.append(item)
                c_indice = tt

                if c_indice != []:
                    indices.extend([item for item in c_indice])#[([2,6], 'for instance'), ....]
                    tagged = tagged.union(set([r for t in c_indice for r in t]))
    return indices
