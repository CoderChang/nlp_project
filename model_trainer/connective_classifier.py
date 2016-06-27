import sys
import json

sys.path.append('..')
import config
import utils.util as util
import utils.conn_util as conn_util
import utils.syntax_tree as Syntax_tree

class connective_classifier(object):

    def __init__(self):
        self.cpos_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_CPOS_PATH)
        self.prev_C_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_PREV_C_PATH)
        self.prevPOS_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_PREVPOS_PATH)
        self.prevPOS_CPOS_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_PREVPOS_CPOS_PATH)
        self.C_next_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_C_NEXT_PATH)
        self.nextPOS_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_NEXTPOS_PATH)
        self.CPOS_nextPOS_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_CPOS_NEXTPOS_PATH)
        self.CParent_to_root_path_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_CPARENT_TO_ROOT_PATH)
        self.compressed_CParent_to_root_path_dict = util.load_dict_from_file(config.CONNECTIVE_DICT_COMPRESSED_CPARENT_TO_ROOT_PATH)

    def train_model(self, pdtb_data_file, pdtb_parses_file):
        with open(pdtb_data_file) as f1:
            data_json_list = [json.loads(line) for line in f1.readlines()]
        with open(pdtb_parses_file) as f2:
            all_parse_dicts = json.loads(f2.read())

        train_examples = []
        for data_json in data_json_list:
            tmp_DocID = data_json['DocID']
            doc = all_parse_dicts[tmp_DocID]
            conn_list = conn_util.get_doc_conns(doc)

            true_conn_indices = []
            if data_json['Type'] == 'Explicit':
                true_conn_indices_begin = data_json['Connective']['TokenList'][0][2]
                true_conn_length = len(data_json['Connective']['RawText'].split(' '))
                true_conn_indices = range(true_conn_indices_begin, true_conn_indices_begin + true_conn_length)

            for (sent_index, conn_indices) in conn_list:
                doc_conn_indices = conn_indices
                for ind, sent in enumerate(doc["sentences"]):
                    if sent_index > ind:
                        doc_conn_indices += len(sent['words'])
                tmp_feature = self.extract_features(doc, sent_index, conn_indices)
                if set(doc_conn_indices).issubset(set(true_conn_indices)):
                    train_examples.append(tmp_feature, True)
                else:
                    train_examples.append(tmp_feature, False)


    def write_model_tofile(self, file_path):
        pass

    def extract_features(self, doc, sent_index, conn_indices):
        # feat dict
        # {
        #     dimension = 1000,
        #     dict_index = 10/None,
        # }
        feat_dict_CPOS_dict = {}
        feat_dict_prev_C_dict = {}
        feat_dict_prevPOS_dict = {}
        feat_dict_prevPOS_CPOS_dict = {}
        feat_dict_C_next_dict = {}
        feat_dict_nextPOS_dict = {}
        feat_dict_CPOS_nextPOS_dict = {}
        feat_dict_CParent_to_root_path_dict = {}
        feat_dict_compressed_CParent_to_root_path_dict = {}

        ''' c pos '''
        pos_tag_list = []
        for conn_index in conn_indices:
            pos_tag_list.append(doc["sentences"][sent_index]["words"][conn_index][1]["PartOfSpeech"])
        CPOS = "_".join(pos_tag_list)

        ''' prev '''
        flag = 0
        prev_index = conn_indices[0] - 1
        prev_sent_index = sent_index
        if prev_index < 0:
            prev_index = -1
            prev_sent_index -= 1
            if prev_sent_index < 0:
                flag = 1

        if flag == 1 :
            prev = "NONE"
        else:
            prev = doc["sentences"][prev_sent_index]["words"][prev_index][0]

        ''' conn_name '''
        conn_name = " ".join([doc["sentences"][sent_index]["words"][word_token][0] \
                    for word_token in conn_indices ])

        '''prevPOS'''
        if prev == "NONE":
            prevPOS = "NONE"
        else:
            prevPOS = doc["sentences"][prev_sent_index]["words"][prev_index][1]["PartOfSpeech"]

        '''next'''
        sent_count = len(doc["sentences"])
        sent_length = len(doc["sentences"][sent_index]["words"])

        flag = 0
        next_index = conn_indices[-1] + 1
        next_sent_index = sent_index
        if next_index >= sent_length:
            next_sent_index += 1
            next_index = 0
            if next_sent_index >= sent_count:
                flag = 1

        if flag == 1:
            next = "NONE"
        else:
            next = doc["sentences"][next_sent_index]["words"][next_index][0]

        ''' next pos '''
        if next == "NONE":
            nextPOS = "NONE"
        else:
            nextPOS = doc["sentences"][next_sent_index]["words"][next_index][1]["PartOfSpeech"]

        parse_tree = doc["sentences"][sent_index]["parsetree"].strip()
        syntax_tree = Syntax_tree(parse_tree)

        ''' c parent to root '''
        if syntax_tree.tree == None:
            cparent_to_root_path = "NONE_TREE"
        else:
            cparent_to_root_path = ""
            for conn_index in conn_indices:
                conn_node = syntax_tree.get_leaf_node_by_token_index(conn_index)
                conn_parent_node = conn_node.up
                cparent_to_root_path += syntax_tree.get_node_path_to_root(conn_parent_node) + "&"
            if cparent_to_root_path[-1] == "&":
                cparent_to_root_path = cparent_to_root_path[:-1]

        ''' compressed c parent to root '''
        if syntax_tree.tree == None:
            compressed_path = "NONE_TREE"
        else:
            compressed_path = ""
            for conn_index in conn_indices:
                conn_node = syntax_tree.get_leaf_node_by_token_index(conn_index)
                conn_parent_node = conn_node.up

                path = syntax_tree.get_node_path_to_root(conn_parent_node)

                compressed_path += util.get_compressed_path(path) + "&"

            if compressed_path[-1] == "&":
                compressed_path = compressed_path[:-1]

        prev_C = "%s|%s" % (prev, conn_name)
        prePOS_CPOS = "%s|%s" % (prevPOS, CPOS)
        C_next = "%s|%s" % (conn_name, next)
        CPOS_nextPOS = "%s|%s" % (CPOS, nextPOS)

        features = []
        features.append(util.get_feature(feat_dict_CPOS_dict, self.cpos_dict, CPOS))
        features.append(util.get_feature(feat_dict_prev_C_dict, self.prev_C_dict, prev_C))
        features.append(util.get_feature(feat_dict_prevPOS_dict, self.prevPOS_dict, prevPOS))
        features.append(util.get_feature(feat_dict_prevPOS_CPOS_dict, self.prevPOS_CPOS_dict, prePOS_CPOS ))
        features.append(util.get_feature(feat_dict_C_next_dict, self.C_next_dict, C_next))
        features.append(util.get_feature(feat_dict_nextPOS_dict, self.nextPOS_dict, nextPOS))
        features.append(util.get_feature(feat_dict_CPOS_nextPOS_dict, self.CPOS_nextPOS_dict, CPOS_nextPOS))
        features.append(util.get_feature(feat_dict_CParent_to_root_path_dict, self.CParent_to_root_path_dict, cparent_to_root_path ))
        features.append(util.get_feature(feat_dict_compressed_CParent_to_root_path_dict, self.compressed_CParent_to_root_path_dict, compressed_path))

        joint_features = util.merge_features(features)
