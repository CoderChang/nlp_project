import sys
import json
import time
import nltk
import pickle

sys.path.append('..')
import config
import utils.util as util
import utils.conn_util as conn_util


class Argument_pos_classifier(object):

    def __init__(self):
        self.classifier = None

        self.dict_CString = util.load_dict_from_file(config.ARG_POSITION_DICT_CSTRING)
        self.dict_CPosition = util.load_dict_from_file(config.ARG_POSITION_DICT_CPOSITION)
        self.dict_CPOS = util.load_dict_from_file(config.ARG_POSITION_DICT_CPOS)
        self.dict_prev1 = util.load_dict_from_file(config.ARG_POSITION_DICT_PREV1)
        self.dict_prev1POS = util.load_dict_from_file(config.ARG_POSITION_DICT_PREV1POS)
        self.dict_prev1_C = util.load_dict_from_file(config.ARG_POSITION_DICT_PREV1_C)
        self.dict_prev1POS_CPOS = util.load_dict_from_file(config.ARG_POSITION_DICT_PREV1POS_CPOS)
        self.dict_prev2 = util.load_dict_from_file(config.ARG_POSITION_DICT_PREV2)
        self.dict_prev2POS = util.load_dict_from_file(config.ARG_POSITION_DICT_PREV2POS)
        self.dict_prev2_C = util.load_dict_from_file(config.ARG_POSITION_DICT_PREV2_C)
        self.dict_prev2POS_CPOS = util.load_dict_from_file(config.ARG_POSITION_DICT_PREV2POS_CPOS)


    def write_model_to_file(self, file_path):
        f = open(file_path, 'wb')
        pickle.dump(self.classifier, f)
        f.close()

    def load_model_from_file(self, file_path):
        f = open(file_path, 'rb')
        self.classifier = pickle.load(f)
        f.close()

    def train_model(self, pdtb_data_file, pdtb_parses_file):
        print 'opening files ...'
        with open(pdtb_data_file) as f1:
            data_json_list = [json.loads(line) for line in f1.readlines()]
        with open(pdtb_parses_file) as f2:
            all_parse_dicts = json.loads(f2.read())

        #train_num = 1000
        #print 'length of data_json_list: ', len(data_json_list), 'train_num: ', train_num
        #data_json_list = data_json_list[:train_num]

        print 'generating train_examples...'
        print time.strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------------'
        PS_num = SS_num = FS_num = 0
        train_examples = []
        for data_json in data_json_list:
            tmp_DocID = data_json['DocID']
            doc = all_parse_dicts[tmp_DocID]
            conn_list = conn_util.get_doc_conns(doc)

            if data_json['Type'] == 'Explicit':
                conn_indices = [token[4] for token in data_json['Connective']['TokenList']]
                data_conn_offset_begin = data_json['Connective']['CharacterSpanList'][0][0]
                data_arg1_offset_begin = data_json['Arg1']['CharacterSpanList'][0][0]
                for ind, sent in enumerate(doc["sentences"]):
                    sent_offset_begin = sent['words'][0][1]['CharacterOffsetBegin']
                    sent_offset_end = sent['words'][-1][1]['CharacterOffsetEnd']
                    if sent_offset_begin <= data_conn_offset_begin <= sent_offset_end:
                        conn_sentence_index = ind
                    if sent_offset_begin <= data_arg1_offset_begin <= sent_offset_end:
                        arg1_sentence_index = ind
                tmp_feature = self.extract_features(doc, conn_sentence_index, conn_indices)
                if arg1_sentence_index == conn_sentence_index:
                    train_examples.append((tmp_feature, 'SS'))
                    SS_num += 1
                elif arg1_sentence_index < conn_sentence_index:
                    train_examples.append((tmp_feature, 'PS'))
                    PS_num += 1
                else:
                    FS_num += 1

        print 'PS_num: ', PS_num, ' SS_num: ', SS_num, ' FS_num: ', FS_num
        print 'train_examples generated, train classifier ...'
        print time.strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------------'

        # MaxentClassifier
        #GIS_algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
        #self.classifier = nltk.MaxentClassifier.train(train_examples, GIS_algorithm, trace=0, max_iter=1000)
        #IIS_algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[1]
        #self.classifier = nltk.MaxentClassifier.train(train_examples, IIS_algorithm, trace=0, max_iter=1000)
        # NaiveBayesClassifier
        self.classifier = nltk.classify.NaiveBayesClassifier.train(train_examples)

        print 'classifier completed ...'
        print time.strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------------'

    # used for prediction
    def get_SSPS_conn_list(self, doc, conn_list):
        PS_conn_list = []
        SS_conn_list = []
        test_features = []
        for (sent_index, conn_indices) in conn_list:
            tmp_feature = self.extract_features(doc, sent_index, conn_indices)
            test_features.append(tmp_feature)
        result_list = self.classifier.classify_many(test_features)
        for i in range(len(result_list)):
            if result_list[i] == 'PS':
                PS_conn_list.append(conn_list[i])
            else:
                SS_conn_list.append(conn_list[i])
        return SS_conn_list, PS_conn_list

    # here, conn_indices are indices within sentence, not within whole doc
    def extract_features(self, doc, sent_index, conn_indices):
        feat_dict_CString = {}
        feat_dict_CPosition = {}
        feat_dict_CPOS = {}
        feat_dict_prev1 = {}
        feat_dict_prev1POS = {}
        feat_dict_prev1_C = {}
        feat_dict_prev1POS_CPOS = {}
        feat_dict_prev2 = {}
        feat_dict_prev2POS = {}
        feat_dict_prev2_C = {}
        feat_dict_prev2POS_CPOS = {}

        '''CString'''
        C_String = " ".join([doc["sentences"][sent_index]["words"][word_token][0] for word_token in conn_indices])

        '''CPosition'''
        sent_length = len(doc["sentences"][sent_index]["words"])
        position = float(conn_indices[0])/float(sent_length)
        if position <= 0.2:
            C_Position = "start"
        elif position >= 0.8:
            C_Position = "end"
        else:
            C_Position = "middle"

        '''CPOS'''
        pos_tag_list = []
        for conn_index in conn_indices:
            pos_tag_list.append(doc["sentences"][sent_index]["words"][conn_index][1]["PartOfSpeech"])
        CPOS = "_".join(pos_tag_list)

        '''prev1'''
        flag = 0
        prev_index = conn_indices[0] - 1
        pre_sent_index = sent_index
        if prev_index < 0:
            pre_sent_index -= 1
            prev_index = -1
            if pre_sent_index < 0:
                flag = 1
        # the previous word of the connective
        if flag == 1:
            prev1 = "prev1_NONE"
        else:
            prev1 = doc["sentences"][pre_sent_index]["words"][prev_index][0]

        '''prev1POS'''
        flag = 0
        prev_index = conn_indices[0] - 1
        pre_sent_index = sent_index
        if prev_index < 0:
            pre_sent_index -= 1
            prev_index = -1
            if pre_sent_index < 0:
                flag = 1
        # the previous word of the connective
        if flag == 1:
            prev1 = "prev1_NONE"
        else:
            prev1 = doc["sentences"][pre_sent_index]["words"][prev_index][0]
        if prev1 == "prev1_NONE":
            prev1POS = "prev1POS_NONE"
        else:
            prev1POS = doc["sentences"][pre_sent_index]["words"][prev_index][1]["PartOfSpeech"]

        '''prev1_C'''
        prev1_C = "%s|%s" % (prev1, C_String)

        '''prev1POS_CPOS'''
        prev1POS_CPOS = "%s|%s" % (prev1POS, CPOS)

        '''prev2'''
        flag = 0
        prev2_index = conn_indices[0] - 2
        pre_sent_index = sent_index
        if prev2_index == -1:
            pre_sent_index -= 1
            if pre_sent_index < 0:
                flag = 1
        elif prev2_index == -2:
            pre_sent_index -= 1
            if pre_sent_index < 0:
                flag = 1
            elif len(doc["sentences"][pre_sent_index]["words"]) == 1:
                pre_sent_index -= 1
                if pre_sent_index < 0:
                    flag = 1
                else:
                    prev2_index = -1
        if flag == 1:
            prev2 = "prev2_NONE"
        else:
            prev2 = doc["sentences"][pre_sent_index]["words"][prev2_index][0]

        '''prev2POS'''
        flag = 0
        prev2_index = conn_indices[0] - 2
        pre_sent_index = sent_index
        if prev2_index == -1:
            pre_sent_index -= 1
            if pre_sent_index < 0:
                flag = 1
        elif prev2_index == -2:
            pre_sent_index -= 1
            if pre_sent_index < 0:
                flag = 1
            elif len(doc["sentences"][pre_sent_index]["words"]) == 1:
                pre_sent_index -= 1
                if pre_sent_index < 0:
                    flag = 1
                else:
                    prev2_index = -1
        if flag == 1:
            prev2 = "prev2_NONE"
        else:
            prev2 = doc["sentences"][pre_sent_index]["words"][prev2_index][0]

        if prev2 == "prev2_NONE":
            prev2POS = "prev2POS_NONE"
        else:
            prev2POS = doc["sentences"][pre_sent_index]["words"][prev2_index][1]["PartOfSpeech"]

        '''prev2_C'''
        prev2_C = "%s|%s" % (prev2, C_String)

        '''prev2POS_CPOS'''
        prev2POS_CPOS = "%s|%s" % (prev2POS, CPOS)

        features = []
        features.append(util.get_feature(feat_dict_CString, self.dict_CString, C_String))
        features.append(util.get_feature(feat_dict_CPosition, self.dict_CPosition, C_Position))
        features.append(util.get_feature(feat_dict_CPOS, self.dict_CPOS, CPOS))
        features.append(util.get_feature(feat_dict_prev1, self.dict_prev1, prev1))
        features.append(util.get_feature(feat_dict_prev1POS, self.dict_prev1POS, prev1POS))
        features.append(util.get_feature(feat_dict_prev1_C, self.dict_prev1_C, prev1_C))
        features.append(util.get_feature(feat_dict_prev1POS_CPOS, self.dict_prev1POS_CPOS, prev1POS_CPOS))
        features.append(util.get_feature(feat_dict_prev2, self.dict_prev2, prev2))
        features.append(util.get_feature(feat_dict_prev2POS, self.dict_prev2POS, prev2POS))
        features.append(util.get_feature(feat_dict_prev2_C, self.dict_prev2_C, prev2_C))
        features.append(util.get_feature(feat_dict_prev2POS_CPOS, self.dict_prev2POS_CPOS, prev2POS_CPOS))

        joint_features = util.merge_features(features)
        #print 'joint_features', joint_features
        return joint_features

    def test_model(self, pdtb_data_file, pdtb_parses_file):
        print 'opening files ...'
        with open(pdtb_data_file) as f1:
            data_json_list = [json.loads(line) for line in f1.readlines()]
        with open(pdtb_parses_file) as f2:
            all_parse_dicts = json.loads(f2.read())

        #test_num = 1000
        #print 'length of data_json_list: ', len(data_json_list), 'test_num: ', test_num
        #data_json_list = data_json_list[:test_num]

        print 'generating test_examples...'
        print time.strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------------'
        train_examples = []
        test_examples = []
        for data_json in data_json_list:
            tmp_DocID = data_json['DocID']
            doc = all_parse_dicts[tmp_DocID]
            conn_list = conn_util.get_doc_conns(doc)

            if data_json['Type'] == 'Explicit':
                conn_indices = [token[4] for token in data_json['Connective']['TokenList']]
                data_conn_offset_begin = data_json['Connective']['CharacterSpanList'][0][0]
                data_arg1_offset_begin = data_json['Arg1']['CharacterSpanList'][0][0]
                for ind, sent in enumerate(doc["sentences"]):
                    sent_offset_begin = sent['words'][0][1]['CharacterOffsetBegin']
                    sent_offset_end = sent['words'][-1][1]['CharacterOffsetEnd']
                    if sent_offset_begin <= data_conn_offset_begin <= sent_offset_end:
                        conn_sentence_index = ind
                    if sent_offset_begin <= data_arg1_offset_begin <= sent_offset_end:
                        arg1_sentence_index = ind
                tmp_feature = self.extract_features(doc, conn_sentence_index, conn_indices)
                test_examples.append(tmp_feature)
                if arg1_sentence_index == conn_sentence_index:
                    train_examples.append((tmp_feature, 'SS'))
                elif arg1_sentence_index < conn_sentence_index:
                    train_examples.append((tmp_feature, 'PS'))

        print 'test_examples generated, test classifier ...'
        print time.strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------------'

        result_list = self.classifier.classify_many(test_examples)
        print 'length of result_list: ', len(result_list)
        true_predict_num = 0
        for i in range(len(result_list)):
            if result_list[i] == train_examples[i][1]:
                true_predict_num += 1
        print 'true_predict_num: ', true_predict_num, ' precision: ', float(true_predict_num)/len(result_list)

        print time.strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------------'


if __name__ == '__main__':
    classifier = Argument_pos_classifier()
    print 'train ...................................................................'
    classifier.train_model(config.TRAIN_DATA_PATH, config.TRAIN_PARSES_PATH)
    classifier.write_model_to_file(config.TRAIN_MODEL_ARG_POS_CL)
    print 'test on training set ....................................................'
    classifier.load_model_from_file(config.TRAIN_MODEL_ARG_POS_CL)
    classifier.test_model(config.TRAIN_DATA_PATH, config.TRAIN_PARSES_PATH)
