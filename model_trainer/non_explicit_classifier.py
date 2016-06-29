import sys
import json
import time
import nltk
import pickle

sys.path.append('..')
import config
import utils.util as util
import utils.conn_util as conn_util

class Non_explicit_classifier(object):

    def __init__(self):
        self.classifier = None

        self.dict_dependency_rules = util.load_dict_from_file(config.NON_EXPLICIT_DICT_DEPENDENCY_RULES)
        self.dict_production_rules = util.load_dict_from_file(config.NON_EXPLICIT_DICT_PRODUCTION_RULES)
        self.dict_brown_cluster = util.load_dict_from_file(config.NON_EXPLICIT_DICT_BROWN_CLUSTER)
        self.dict_word_pairs = util.load_dict_from_file(config.NON_EXPLICIT_DICT_WORD_PAIRS)

        self.non_explicit_type_list = [
            'Implicit',
            'EntRel',
            'AltLex'
        ]

        self.implicit_sense_list = [
            'Expansion.Conjunction',
            'Expansion.Restatement',
            'Contingency.Cause.Reason',
            'Comparison.Contrast',
            'Contingency.Cause.Result',
            'Expansion.Instantiation',
            'Temporal.Asynchronous.Precedence',
            'Temporal.Synchrony',
            'Comparison.Concession',
            'Comparison',
            'EntRel'
        ]


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
        Sense_statistics, Type_statistics = {}, {}
        train_examples = []
        for data_json in data_json_list:
            tmp_DocID = data_json['DocID']
            doc = all_parse_dicts[tmp_DocID]

            if data_json['Type'] != 'Explicit':
                relation = data_json
                tmp_feature = self.extract_features(doc, relation)
                for tmp_sense in relation['Sense']:
                    if tmp_sense in self.implicit_sense_list:
                        train_examples.append((tmp_feature, tmp_sense))

                    if tmp_sense in Sense_statistics :
                        Sense_statistics[tmp_sense] += 1
                    else :
                        Sense_statistics[tmp_sense] = 1

                if relation['Type'] in Type_statistics :
                    Type_statistics[relation['Type']] += 1
                else :
                    Type_statistics[relation['Type']] = 1

        for k in Sense_statistics :
            print k + ': ' + str(Sense_statistics[k])
        for k in Type_statistics :
            print k + ': ' + str(Type_statistics[k])
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
    def get_non_explicit_relations(doc, non_explicit_relations):
        return non_explicit_relations

    # used for prediction
    def divide_non_explicit_relations(non_explicit_relations, doc):
        EntRel_relations = []
        Implicit_AltLex_relations = []

        return EntRel_relations, Implicit_AltLex_relations


    def extract_features(self, doc, relation):
        feat_dict_arg1_dependency = {}
        feat_dict_arg2_dependency = {}
        feat_dict_arg1_and_arg2_dependency = {}
        feat_dict_production = {}
        feat_dict_cluster = {}
        feat_dict_word_pairs = {}

        '''dependency rules'''
        Arg1_dependency_rules = conn_util.get_Arg_dependency_rules(relation, "Arg1", doc)
        Arg2_dependency_rules = conn_util.get_Arg_dependency_rules(relation, "Arg2", doc)
        Arg1_and_Arg2_dependency_rules = list(set(Arg1_dependency_rules) & set(Arg2_dependency_rules))

        '''production rules'''
        Arg1_production_rules = conn_util.get_Arg_production_rules(relation, "Arg1", doc)
        Arg2_production_rules = conn_util.get_Arg_production_rules(relation, "Arg2", doc)
        Arg1_and_Arg2_production_rules = list(set(Arg1_production_rules) & set(Arg2_production_rules))

        Arg1_production_rules = ["Arg1_%s" % rule for rule in Arg1_production_rules]
        Arg2_production_rules = ["Arg2_%s" % rule for rule in Arg2_production_rules]
        Both_production_rules = ["Both_%s" % rule for rule in Arg1_and_Arg2_production_rules]
        production_rules = Arg1_production_rules + Arg2_production_rules + Both_production_rules

        '''brown cluster'''
        Arg1_brown_cluster = conn_util.get_Arg_brown_cluster(relation, "Arg1", doc)
        Arg2_brown_cluster = conn_util.get_Arg_brown_cluster(relation, "Arg2", doc)
        Both_brown_cluster = list(set(Arg1_brown_cluster) & set(Arg2_brown_cluster))

        Arg1_only = list(set(Arg1_brown_cluster) - set(Arg2_brown_cluster))
        Arg2_only = list(set(Arg2_brown_cluster) - set(Arg1_brown_cluster))

        Arg1_brown_cluster = ["Arg1_%s" % x for x in Arg1_only]
        Arg2_brown_cluster = ["Arg2_%s" % x for x in Arg2_only]
        Both_brown_cluster = ["Both_%s" % x for x in Both_brown_cluster]
        brown_cluster = Arg1_brown_cluster + Arg2_brown_cluster + Both_brown_cluster

        '''word pairs'''
        word_pairs = conn_util.get_word_pairs(relation, doc)

        features = []
        features.append(util.get_feature_by_list(feat_dict_arg1_dependency, self.dict_dependency_rules, Arg1_dependency_rules))
        features.append(util.get_feature_by_list(feat_dict_arg2_dependency, self.dict_dependency_rules, Arg2_dependency_rules))
        features.append(util.get_feature_by_list(feat_dict_arg1_and_arg2_dependency, self.dict_dependency_rules, Arg1_and_Arg2_dependency_rules))
        features.append(util.get_feature_by_list(feat_dict_production, self.dict_production_rules, production_rules))
        features.append(util.get_feature_by_list(feat_dict_cluster, self.dict_brown_cluster, brown_cluster))
        features.append(util.get_feature_by_list(feat_dict_word_pairs, self.dict_word_pairs, word_pairs))

        joint_features = util.merge_features(features)
        #print 'joint_features', joint_features
        return joint_features

    def test_model(self, pdtb_data_file, pdtb_parses_file):
        print 'opening files ...'
        with open(pdtb_data_file) as f1:
            data_json_list = [json.loads(line) for line in f1.readlines()]
        with open(pdtb_parses_file) as f2:
            all_parse_dicts = json.loads(f2.read())

        test_num = 1000
        print 'length of data_json_list: ', len(data_json_list), 'test_num: ', test_num
        data_json_list = data_json_list[:test_num]

        print 'generating test_examples...'
        print time.strftime('%Y-%m-%d %H:%M:%S')
        print '------------------------------------------'

        train_examples = []
        test_examples = []
        for data_json in data_json_list:
            tmp_DocID = data_json['DocID']
            doc = all_parse_dicts[tmp_DocID]

            if data_json['Type'] != 'Explicit':
                relation = data_json
                tmp_feature = self.extract_features(doc, relation)
                for tmp_sense in relation['Sense']:
                    train_examples.append((tmp_feature, tmp_sense))
                    test_examples.append(tmp_feature)

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
    classifier = Non_explicit_classifier()
    print 'train ...................................................................'
    classifier.train_model(config.TRAIN_DATA_PATH, config.TRAIN_PARSES_PATH)
    classifier.write_model_to_file(config.TRAIN_MODEL_NON_EXPLICIT_CL)
    print 'test on training set ....................................................'
    classifier.load_model_from_file(config.TRAIN_MODEL_NON_EXPLICIT_CL)
    classifier.test_model(config.TRAIN_DATA_PATH, config.TRAIN_PARSES_PATH)
