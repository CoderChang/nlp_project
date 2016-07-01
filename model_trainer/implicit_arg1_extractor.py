import sys
import os
import json
import time
import nltk
import pickle

sys.path.append('..')
import config
import utils.util as util
import utils.conn_util as conn_util

WRITE_EXAMPLE_FLAG = False

class Implicit_arg1_extractor(object):

    def __init__(self):
        self.classifier = None

        self.dict_prev_curr_CP_production_rule = util.load_dict_from_file(config.ATTRIBUTION_NON_CONNS_DICT_PREV_CURR_CP_PRODUCTION_RULE)
        self.dict_curr_last = util.load_dict_from_file(config.ATTRIBUTION_NON_CONNS_DICT_CURR_LAST)
        self.dict_lemma_verbs = util.load_dict_from_file(config.ATTRIBUTION_NON_CONNS_DICT_LEMMA_VERBS)
        self.dict_prev2_pos_lemma_verb = util.load_dict_from_file(config.ATTRIBUTION_NON_CONNS_DICT_2PREV_POS_LEMMA_VERB)
        self.dict_is_curr_NNP_prev_PRP_or_NNP = {"NONE": 1, "yes": 2, "no": 2}
        self.dict_clause_word_num = {}
        for i in range(200):
            self.dict_clause_word_num[i] = i

    def write_model_to_file(self, file_path):
        f = open(file_path, 'wb')
        pickle.dump(self.classifier, f)
        f.close()

    def load_model_from_file(self, file_path):
        f = open(file_path, 'rb')
        self.classifier = pickle.load(f)
        f.close()

    def train_model(self, pdtb_data_file, pdtb_parses_file):
        if WRITE_EXAMPLE_FLAG and os.path.exists(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TRAIN):
            print 'using existed training examples ...'
            train_examples = util.load_examples_from_file(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TRAIN)
            print 'training examples loaded.'
        else:
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
            train_examples = []
            for data_json in data_json_list:
                tmp_DocID = data_json['DocID']
                doc = all_parse_dicts[tmp_DocID]

                if data_json['Type'] != 'Explicit':
                    relation = data_json
                    true_arg_indices = [token[4] for token in relation['Arg1']['TokenList']]
                    sent_index = relation['Arg1']['TokenList'][0][3]
                    for arg_clauses in conn_util.get_sentence_clauses(doc, sent_index):
                        if arg_clauses == []:
                            continue
                        for clause_index in range(len(arg_clauses.clauses)):
                            tmp_feature = self.extract_features(arg_clauses, clause_index, doc)
                            curr_clause_indices = arg_clauses.clauses[clause_index][0]
                            if set(curr_clause_indices) <= set(true_arg_indices):
                                train_examples.append((tmp_feature, 'yes'))
                            else:
                                train_examples.append((tmp_feature, 'no'))

            if WRITE_EXAMPLE_FLAG:
                util.write_examples_to_file(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TRAIN, train_examples)

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
    def extract_argument(self, doc, relation):
        arg1_token_list = []
        '''extract clauses'''
        for arg_clauses in conn_util.get_arg1_clauses(doc, relation):
            if arg_clauses == []:
                continue
            for clause_index in range(len(arg_clauses.clauses)):
                tmp_feature = self.extract_features(arg_clauses, clause_index, doc)
                result = self.classifier.classify(tmp_feature)
                if result == 'yes':
                    arg1_token_list.extend(arg_clauses.clauses[clause_index][0])
        relation['Arg1']['TokenList'] = arg1_token_list
        return relation

    def extract_features(self, arg_clauses, clause_index, doc):
        feat_dict_prev_curr_CP_production_rule = {}
        feat_dict_curr_last = {}
        feat_dict_lemma_verbs = {}
        feat_dict_prev2_pos_lemma_verb = {}
        feat_dict_clause_word_num = {}
        feat_dict_is_curr_NNP_prev_PRP_or_NNP = {}

        '''pre_curr_CP_production_rule'''
        prev_curr_CP_production_rule = conn_util.get_prev_curr_CP_production_rule(arg_clauses, clause_index, doc)

        '''curr_last'''
        curr_last = conn_util.get_curr_last(arg_clauses, clause_index, doc)

        '''lemma_verbs_list'''
        lemma_verbs_list = conn_util.get_curr_lemma_verbs(arg_clauses, clause_index, doc)

        '''prev2_pos_lemma_verb'''
        prev2_pos_lemma_verb = conn_util.get_2prev_pos_lemma_verb(arg_clauses, clause_index, doc)

        '''clause_word_num'''
        clause_word_num = len(arg_clauses.clauses[clause_index][0])

        '''is_curr_NNP_prev_PRP_or_NNP'''
        is_curr_NNP_prev_PRP_or_NNP = conn_util.get_is_curr_NNP_prev_PRP_or_NNP(arg_clauses, clause_index, doc)

        features = []
        features.append(util.get_feature_by_list(feat_dict_prev_curr_CP_production_rule, self.dict_prev_curr_CP_production_rule, prev_curr_CP_production_rule))
        features.append(util.get_feature_by_list(feat_dict_curr_last, self.dict_curr_last, curr_last))
        features.append(util.get_feature_by_list(feat_dict_lemma_verbs, self.dict_lemma_verbs, lemma_verbs_list))
        features.append(util.get_feature(feat_dict_prev2_pos_lemma_verb, self.dict_prev2_pos_lemma_verb, prev2_pos_lemma_verb))
        features.append(util.get_feature(feat_dict_clause_word_num, self.dict_clause_word_num, clause_word_num))
        features.append(util.get_feature(feat_dict_is_curr_NNP_prev_PRP_or_NNP, self.dict_is_curr_NNP_prev_PRP_or_NNP, is_curr_NNP_prev_PRP_or_NNP))

        joint_features = util.merge_features(features)
        #print 'joint_features', joint_features
        return joint_features

    def test_model(self, pdtb_data_file, pdtb_parses_file):
        if WRITE_EXAMPLE_FLAG and os.path.exists(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TEST_TEST) and os.path.exists(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TEST_TRAIN):
            print 'using existed test examples ...'
            test_examples = util.load_examples_from_file(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TEST_TEST)
            test_train_examples = util.load_examples_from_file(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TEST_TRAIN)
            print 'test examples loaded.'
        else:
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
            test_train_examples = []
            test_examples = []
            for data_json in data_json_list:
                tmp_DocID = data_json['DocID']
                doc = all_parse_dicts[tmp_DocID]

                if data_json['Type'] != 'Explicit':
                    relation = data_json
                    true_arg_indices = [token[4] for token in relation['Arg1']['TokenList']]
                    sent_index = relation['Arg1']['TokenList'][0][3]
                    for arg_clauses in conn_util.get_sentence_clauses(doc, sent_index):
                        if arg_clauses == []:
                            continue
                        for clause_index in range(len(arg_clauses.clauses)):
                            tmp_feature = self.extract_features(arg_clauses, clause_index, doc)
                            curr_clause_indices = arg_clauses.clauses[clause_index][0]
                            if set(curr_clause_indices) <= set(true_arg_indices):
                                test_train_examples.append((tmp_feature, 'yes'))
                                test_examples.append(tmp_feature)
                            else:
                                test_train_examples.append((tmp_feature, 'no'))
                                test_examples.append(tmp_feature)

            if WRITE_EXAMPLE_FLAG:
                util.write_examples_to_file(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TEST_TEST, test_examples)
                util.write_examples_to_file(config.TRAIN_EXAMPLES_IMPLICIT_ARG1_TEST_TRAIN, test_train_examples)
            print 'test_examples generated, test classifier ...'
            print time.strftime('%Y-%m-%d %H:%M:%S')
            print '------------------------------------------'

        result_list = self.classifier.classify_many(test_examples)
        print 'length of result_list: ', len(result_list)
        true_predict_num = 0
        for i in range(len(result_list)):
            if result_list[i] == test_train_examples[i][1]:
                true_predict_num += 1
        print 'true_predict_num: ', true_predict_num, ' accuracy: ', float(true_predict_num)/len(result_list)

        print time.strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------------'


if __name__ == '__main__':
    classifier = Implicit_arg1_extractor()
    print 'train ...................................................................'
    classifier.train_model(config.TRAIN_DATA_PATH, config.TRAIN_PARSES_PATH)
    classifier.write_model_to_file(config.TRAIN_MODEL_IMPLICIT_ARG1_EXTRACTOR)
    print 'test on training set ....................................................'
    classifier.load_model_from_file(config.TRAIN_MODEL_IMPLICIT_ARG1_EXTRACTOR)
    classifier.test_model(config.TRAIN_DATA_PATH, config.TRAIN_PARSES_PATH)

