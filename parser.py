import json
import sys
import utils.conn_util as conn_util
from model_trainer.connective_classifier import Connective_classifier
from model_trainer.argument_pos_classifier import Argument_pos_classifier
from model_trainer.non_explicit_classifier import Non_explicit_classifier

class DiscourseParser(object):

    def __init__(self):
        pass

    def parse_file(self, input_file):
        documents = json.loads(open(input_file).read())
        relations = []
        for doc_id in documents:
            relations.extend(self.parse_doc(documents[doc_id], doc_id))
        return relations

    def parse_doc(self, doc, doc_id):
        # my code
        explicit_relations = []
        non_explicit_relations = []
        conn_list = conn_util.get_doc_conns(doc)

        print "==> 1. Connective classifier"
        conn_classifier = Connective_classifier()
        conn_classifier.load_model_from_file(config.TRAIN_MODEL_CONN_CL)
        conn_list = conn_classifier.get_true_conn_list(doc, conn_list)

        print "==> 2. Argument position classifier"
        arg_pos_classifier = Argument_pos_classifier()
        arg_pos_classifier.load_model_from_file(config.TRAIN_MODEL_ARG_POS_CL)
        SS_conn_list, PS_conn_list = arg_pos_classifier.get_SSPS_conn_list(doc, conn_list)

        print "==> 3. Non-Explicit classifier"
        # obtain all adjacent sentence pairs within each paragraph, but not identified in any Explicit relation
        adjacent_non_exp_list = conn_util.get_adjacent_non_exp_list(doc, PS_conn_list)
        # obtain non_explicit relation object list by adjacent_non_exp_list, no sense.
        non_explicit_relations = conn_util.fake_non_explicit_relations(doc_id, doc, adjacent_non_exp_list)

        non_exp_classifier = Non_explicit_classifier()
        non_exp_classifier.load_model_from_file(config.TRAIN_MODEL_NON_EXPLICIT_CL)

        non_explicit_relations = non_exp_classifier.get_non_explicit_relations(doc, non_explicit_relations)
        EntRel_relations, Implicit_AltLex_relations = non_exp_classifier.divide_non_explicit_relations(non_explicit_relations, doc)


        output = explicit_relations + non_explicit_relations

        # sample code
        output = []
        num_sentences = len(doc['sentences'])
        token_id = 0
        for i in range(num_sentences-1):
            sentence1 = doc['sentences'][i]
            len_sentence1 = len(sentence1['words'])
            token_id += len_sentence1
            sentence2 = doc['sentences'][i+1]
            len_sentence2 = len(sentence2['words'])

            relation = {}
            relation['DocID'] = doc_id
            relation['Arg1'] = {}
            relation['Arg1']['TokenList'] = range((token_id - len_sentence1), token_id - 1)
            relation['Arg2'] = {}
            relation['Arg2']['TokenList'] = range(token_id, (token_id + len_sentence2) - 1)
            relation['Type'] = 'Implicit'
            relation['Sense'] = ['Expansion.Conjunction']
            relation['Connective'] = {}
            relation['Connective']['TokenList'] = []
            output.append(relation)
        return output

if __name__ == '__main__':
    input_dataset = sys.argv[1]
    input_run = sys.argv[2]
    output_dir = sys.argv[3]
    parser = DiscourseParser()
    relations = parser.parse_file('%s/pdtb-parses-dev.json' % input_dataset)
    output = open('%s/output.json' % output_dir, 'w')
    for relation in relations:
        output.write('%s\n' % json.dumps(relation))
    output.close()
