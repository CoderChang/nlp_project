import sys
import util
from syntax_tree import Syntax_tree

sys.path.append('..')
import config


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


def get_adjacent_non_exp_list(doc, PS_conn_list):
    exp_rel_sent_pairs = [] # [(1,2),(8,9)...]
    for (sent_index, conn_indices) in PS_conn_list:
        if sent_index == 0:
            continue
        exp_rel_sent_pairs.append((sent_index - 1, sent_index))
    exp_rel_sent_pairs = set(exp_rel_sent_pairs)

    #[(sent1_index,sent2_index),...]
    sent_count = len(doc["sentences"])
    adj_pair_set = _get_adj_pair_set(sent_count)
    adj_non_exp_pair_set = adj_pair_set - exp_rel_sent_pairs
    adjacent_non_exp_list = list(adj_non_exp_pair_set)
    return adjacent_non_exp_list


# [(0, 1), (1, 2), (2, 3), (3, 4)]
def _get_adj_pair_set(length):
    i = 0
    list = []
    while i < length -1:
        list.append((i, i+1))
        i += 1
    return set(list)


def fake_non_explicit_relations(doc_id, doc, adjacent_non_exp_list):
    non_explicit_relations = []
    for index, (sent1_index, sent2_index) in enumerate(adjacent_non_exp_list):
        Arg1_offset_in_sent = _non_explicit_Arg_offset_in_sent(doc, sent1_index)
        Arg2_offset_in_sent = _non_explicit_Arg_offset_in_sent(doc, sent2_index)

        Arg1_TokenList = [ [-1, -1, -1, sent1_index, offset] for offset in Arg1_offset_in_sent]
        Arg2_TokenList = [ [-1, -1, -1, sent2_index, offset] for offset in Arg2_offset_in_sent]

        relation = {}
        relation["ID"] = index
        relation['DocID'] = doc_id
        relation['Arg1'] = {}
        relation['Arg1']['TokenList'] = Arg1_TokenList
        relation['Arg2'] = {}
        relation['Arg2']['TokenList'] = Arg2_TokenList
        relation['Type'] = 'Implicit'
        relation['Connective'] = {}
        relation['Connective']['TokenList'] = []
        non_explicit_relations.append(relation)
    return non_explicit_relations


def divide_non_explicit_relations(non_explicit_relations, doc):
    EntRel_relations = []
    Implicit_AltLex_relations = []
    for relation in non_explicit_relations :
        if relation['Sense'][0] == "EntRel":
            DocID = relation["DocID"]
            Arg1_offset_in_sent = [item[4] for item in relation["Arg1"]["TokenList"]]
            Arg2_offset_in_sent = [item[4] for item in relation["Arg2"]["TokenList"]]
            Arg1_sent_index = relation["Arg1"]["TokenList"][0][3]
            Arg2_sent_index = relation["Arg2"]["TokenList"][0][3]
            relation['Arg1']['TokenList'] = get_doc_offset(doc, Arg1_sent_index, Arg1_offset_in_sent)
            relation['Arg2']['TokenList'] = get_doc_offset(doc, Arg2_sent_index, Arg2_offset_in_sent)
            EntRel_relations.append(relation)
        else:
            Implicit_AltLex_relations.append(relation)
    # print "EntRel_relations:" + str(len(EntRel_relations))
    # print "Implicit_AltLex_relations:" + str(len(Implicit_AltLex_relations))
    return EntRel_relations, Implicit_AltLex_relations


def get_doc_offset(doc, sent_index, list):
    offset = 0
    for i in range(sent_index):
        offset += len(doc["sentences"][i]["words"])
    temp = []
    for item in list:
        temp.append(item + offset)
    return temp


def _non_explicit_Arg_offset_in_sent(doc, sent_index):
    curr_length = len(doc["sentences"][sent_index]["words"])
    Arg = [(index, doc["sentences"][sent_index]["words"][index][0]) for index in range(0, curr_length)]
    Arg = util.list_strip_punctuation(Arg)
    Arg = [item[0] for item in Arg]
    return Arg


def get_Arg_dependency_rules(relation, Arg, doc):
    #1.  dict[sent_index] = [token_list]
    dict = {}
    Arg_TokenList = get_Arg_TokenList(relation, Arg)
    for sent_index, word_index in Arg_TokenList:
        if sent_index not in dict:
            dict[sent_index] = [word_index]
        else:
            dict[sent_index].append(word_index)

    #2. dependency_rules
    dependency_rules = []
    for sent_index in dict:
        Arg_indices = [item+1 for item in dict[sent_index]] #dependency start from index 1
        dependency_list = doc["sentences"][sent_index]["dependencies"]

        depen_dict = {} # depen_dict["talk"] = ["nsubj", "aux"]
        for dependency in dependency_list:
            if int(dependency[1].split("-")[-1]) in Arg_indices:
                word = "-".join(dependency[1].split("-")[:-1])
                if word not in depen_dict:
                    depen_dict[word] = [dependency[0]]
                else:
                    depen_dict[word].append(dependency[0])
        for key in depen_dict:
            rule = key + "<--" + " ".join(depen_dict[key])
            dependency_rules.append(rule)
        # print dependency_rules
    return dependency_rules


#[(sent_index, index)]
# Arg : Arg1 or Arg2
def get_Arg_TokenList(relation, Arg):
    return [(item[3], item[4]) for item in relation[Arg]["TokenList"]]


def get_Arg_production_rules(relation, Arg, doc):
    #1.  dict[sent_index] = [token_list]
    dict = {}
    Arg_TokenList = get_Arg_TokenList(relation, Arg)
    for sent_index, word_index in Arg_TokenList:
        if sent_index not in dict:
            dict[sent_index] = [word_index]
        else:
            dict[sent_index].append(word_index)

    #2. production_rules
    Arg_subtrees = []
    for sent_index in dict.keys():
        parse_tree = doc["sentences"][sent_index]["parsetree"].strip()
        syntax_tree = Syntax_tree(parse_tree)
        if syntax_tree.tree != None:
            Arg_indices = dict[sent_index]
            Arg_leaves = set([syntax_tree.get_leaf_node_by_token_index(index) for index in Arg_indices])
            Arg_leaves_labels = set([leaf.label() for leaf in Arg_leaves])
            for nodeposition in syntax_tree.tree.treepositions():
                node = syntax_tree.tree[nodeposition]
                if set(node.leaves()) <= Arg_leaves_labels:
                    Arg_subtrees.append(node)

    production_rules = []
    for node in Arg_subtrees:
        if not isinstance(node, str):
            rule = node.label() + '-->' + ' '.join([child.label() for child in node])
            production_rules.append(rule)

    production_rules = list(set(production_rules))
    return production_rules


def get_brown_cluster():
    dict = {}
    fin = open(config.BROWN_CLUSTER_PATH)
    for line in fin:
        c, w = line.strip().split("\t")
        dict[w] = c
    fin.close()
    return dict


def get_Arg_Words_List(relation, Arg, doc):
    words = []
    Arg_TokenList = get_Arg_TokenList(relation, Arg)
    for sent_index, word_index in Arg_TokenList:
        word = doc["sentences"][sent_index]["words"][word_index][0]
        words.append(word)
    return words


def get_Arg_brown_cluster(relation, Arg, doc):
    Arg_words = get_Arg_Words_List(relation, Arg, doc)
    dict_brown_cluster = get_brown_cluster()
    Arg_brown_cluster = []
    for word in Arg_words:
        if word in dict_brown_cluster:
            Arg_brown_cluster.append(dict_brown_cluster[word])
    return Arg_brown_cluster


def get_word_pairs(relation, doc):
    Arg1_words = get_Arg_Words_List(relation, "Arg1", doc)
    Arg2_words = get_Arg_Words_List(relation, "Arg2", doc)

    #stem
    Arg1_words = util.stem_list(Arg1_words)
    Arg2_words = util.stem_list(Arg2_words)

    word_pairs = []
    for word1 in Arg1_words:
        for word2 in Arg2_words:
            word_pairs.append("%s|%s" % (word1, word2))
    return word_pairs


