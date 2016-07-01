import json
import sys
import config
import utils.conn_util as conn_util
from model_trainer.connective_classifier import Connective_classifier
from model_trainer.argument_pos_classifier import Argument_pos_classifier
from model_trainer.non_explicit_classifier import Non_explicit_classifier
#from model_trainer.implicit_arg1_extractor import Implicit_arg1_extractor
#from model_trainer.implicit_arg2_extractor import Implicit_arg2_extractor

class DiscourseParser(object):

    def __init__(self):
        print "==> Loading Connective classifier"
        self.conn_classifier = Connective_classifier()
        self.conn_classifier.load_model_from_file(config.TRAIN_MODEL_CONN_CL)
        print "==> Loading Argument position classifier"
        self.arg_pos_classifier = Argument_pos_classifier()
        self.arg_pos_classifier.load_model_from_file(config.TRAIN_MODEL_ARG_POS_CL)
        print "==> Loading Implicit arg1 extractor"
        self.implicit_arg1_extractor = Implicit_arg1_extractor()
        self.implicit_arg1_extractor.load_model_from_file(config.TRAIN_MODEL_IMPLICIT_ARG1_EXTRACTOR)
        print "==> Loading Implicit arg2 extractor"
        self.implicit_arg2_extractor = Implicit_arg2_extractor()
        self.implicit_arg2_extractor.load_model_from_file(config.TRAIN_MODEL_IMPLICIT_ARG2_EXTRACTOR)
        print "==> Loading Non-Explicit classifier"
        self.non_exp_classifier = Non_explicit_classifier()
        self.non_exp_classifier.load_model_from_file(config.TRAIN_MODEL_NON_EXPLICIT_CL)
        print "------------- All models loaded -------------"

    def parse_file(self, input_file):
        documents = json.loads(open(input_file).read())
        relations = []
        for doc_id in documents:
            relations.extend(self.parse_doc(documents[doc_id], doc_id))
        return relations

    def parse_doc(self, doc, doc_id):
        print "Parsing doc: " + doc_id
        non_explicit_relations = []
        conn_list = conn_util.get_doc_conns(doc)

        """Connective classifier"""
        conn_list = self.conn_classifier.get_true_conn_list(doc, conn_list)

        """Argument position classifier"""
        SS_conn_list, PS_conn_list = self.arg_pos_classifier.get_SSPS_conn_list(doc, conn_list)

        # obtain all adjacent sentence pairs within each paragraph, but not identified in any Explicit relation
        adjacent_non_exp_list = conn_util.get_adjacent_non_exp_list(doc, PS_conn_list)
        # fake non_explicit relation object list by adjacent_non_exp_list, no sense.
        non_explicit_relations = conn_util.fake_non_explicit_relations(doc_id, doc, adjacent_non_exp_list)

        """Implicit arg1 extractor"""
        non_explicit_relations = self.implicit_arg1_extractor.extract_argument(doc, non_explicit_relations)

        """Implicit arg2 extractor"""
        non_explicit_relations = self.implicit_arg2_extractor.extract_argument(doc, non_explicit_relations)

        """Non-Explicit classifier"""
        non_explicit_relations = self.non_exp_classifier.get_non_explicit_relations(doc, non_explicit_relations)
        #EntRel_relations, Implicit_AltLex_relations = conn_util.divide_non_explicit_relations(non_explicit_relations, doc)
        #EntRel_relations, Implicit_AltLex_relations = conn_util.fake_divide_non_explicit_relations(non_explicit_relations, doc)

        #output = EntRel_relations + Implicit_AltLex_relations
        output = non_explicit_relations
        print 'Output non-explicit relations num: ', len(output)
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
