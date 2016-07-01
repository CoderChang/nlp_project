CWD = '/home/slhome/cc001/Documents/nlp/nlp_project/'
#CWD = '/home/cc001/Documents/nlp_project/'

# directories:
DATA_DIR = CWD + 'data/'

TRAIN_DIR = CWD + 'train_dataset/'
TEST_DIR = CWD + 'test_dataset/'
MODEL_TRAINER_DIR = CWD + 'model_trainer/'
DICT_DIR = CWD + 'dict/'

TRAIN_OUT_EXAMPLE_DIR = CWD + 'train_output/examples/'
TRAIN_OUT_MODEL_DIR = CWD + 'train_output/model/'

DICT_CONN_DIR = DICT_DIR + 'connective/'
DICT_ARG_POS_DIR = DICT_DIR + 'argument_position/'
DICT_NON_EXPLICIT_DIR = DICT_DIR + "non_explicit_classifier/"
DICT_ATTRIBUTION_NON_CONNS_DIR = DICT_DIR + "implicit_arg/"

# file paths:
TRAIN_DATA_PATH = TRAIN_DIR + 'pdtb-data.json'
TRAIN_PARSES_PATH = TRAIN_DIR + 'pdtb-parses.json'

TRAIN_MODEL_CONN_CL = TRAIN_OUT_MODEL_DIR + 'conn_cl.pickle'
TRAIN_MODEL_ARG_POS_CL = TRAIN_OUT_MODEL_DIR + 'arg_pos_cl.pickle'
TRAIN_MODEL_NON_EXPLICIT_CL = TRAIN_OUT_MODEL_DIR + 'non_exp_cl.pickle'
TRAIN_MODEL_IMPLICIT_ARG1_EXTRACTOR = TRAIN_OUT_MODEL_DIR + 'implicit_arg1_extractor.pickle'
TRAIN_MODEL_IMPLICIT_ARG2_EXTRACTOR = TRAIN_OUT_MODEL_DIR + 'implicit_arg2_extractor.pickle'

TRAIN_EXAMPLES_NON_EXPLICIT_TRAIN = TRAIN_OUT_EXAMPLE_DIR + 'non_exp_examples_train.pickle'
TRAIN_EXAMPLES_NON_EXPLICIT_TEST_TRAIN = TRAIN_OUT_EXAMPLE_DIR + 'non_exp_examples_test_train.pickle'
TRAIN_EXAMPLES_NON_EXPLICIT_TEST_TEST = TRAIN_OUT_EXAMPLE_DIR + 'non_exp_examples_test_test.pickle'

TRAIN_EXAMPLES_IMPLICIT_ARG1_TRAIN = TRAIN_OUT_EXAMPLE_DIR + 'implicit_arg1_examples_train.pickle'
TRAIN_EXAMPLES_IMPLICIT_ARG1_TEST_TRAIN = TRAIN_OUT_EXAMPLE_DIR + 'implicit_arg1_examples_test_train.pickle'
TRAIN_EXAMPLES_IMPLICIT_ARG1_TEST_TEST = TRAIN_OUT_EXAMPLE_DIR + 'implicit_arg1_examples_test_test.pickle'

TRAIN_EXAMPLES_IMPLICIT_ARG2_TRAIN = TRAIN_OUT_EXAMPLE_DIR + 'implicit_arg2_examples_train.pickle'
TRAIN_EXAMPLES_IMPLICIT_ARG2_TEST_TRAIN = TRAIN_OUT_EXAMPLE_DIR + 'implicit_arg2_examples_test_train.pickle'
TRAIN_EXAMPLES_IMPLICIT_ARG2_TEST_TEST = TRAIN_OUT_EXAMPLE_DIR + 'implicit_arg2_examples_test_test.pickle'

SORTED_EXP_CONN_PATH = DATA_DIR + 'sortedExpConn.txt'
BROWN_CLUSTER_PATH = DATA_DIR + 'brown_cluster_3200.txt'

''' connnective dict '''
CONNECTIVE_DICT_CPOS_PATH = DICT_CONN_DIR + "cpos_dict.txt"
CONNECTIVE_DICT_PREV_C_PATH = DICT_CONN_DIR +"prev_c_dict.txt"
CONNECTIVE_DICT_PREVPOS_PATH = DICT_CONN_DIR +"prevpos_dict.txt"
CONNECTIVE_DICT_PREVPOS_CPOS_PATH = DICT_CONN_DIR + "prevpos_cpos_dict.txt"
CONNECTIVE_DICT_C_NEXT_PATH = DICT_CONN_DIR + "c_next_dict.txt"
CONNECTIVE_DICT_NEXTPOS_PATH = DICT_CONN_DIR + "nextpos_dict.txt"
CONNECTIVE_DICT_CPOS_NEXTPOS_PATH = DICT_CONN_DIR + "cpos_nextpos_dict.txt"
CONNECTIVE_DICT_CPARENT_TO_ROOT_PATH = DICT_CONN_DIR+ "cparent_to_root_path_dict.txt"
CONNECTIVE_DICT_COMPRESSED_CPARENT_TO_ROOT_PATH = DICT_CONN_DIR+ "compressed_cparent_to_root_path_dict.txt"

''' argument position dict '''
ARG_POSITION_DICT_CSTRING = DICT_ARG_POS_DIR + "ctring_dict.txt"
ARG_POSITION_DICT_CPOSITION = DICT_ARG_POS_DIR + "cposition_dict.txt"
ARG_POSITION_DICT_CPOS = DICT_ARG_POS_DIR + "cpos_dict.txt"
ARG_POSITION_DICT_PREV1 = DICT_ARG_POS_DIR + "prev1_dict.txt"
ARG_POSITION_DICT_PREV1POS = DICT_ARG_POS_DIR + "prev1pos_dict.txt"
ARG_POSITION_DICT_PREV1_C = DICT_ARG_POS_DIR + "prev1_C_dict.txt"
ARG_POSITION_DICT_PREV1POS_CPOS = DICT_ARG_POS_DIR + "prev1pos_Cpos_dict.txt"
ARG_POSITION_DICT_PREV2 = DICT_ARG_POS_DIR + "prev2_dict.txt"
ARG_POSITION_DICT_PREV2POS = DICT_ARG_POS_DIR + "prev2pos_dict.txt"
ARG_POSITION_DICT_PREV2_C = DICT_ARG_POS_DIR + "prev2_C_dict.txt"
ARG_POSITION_DICT_PREV2POS_CPOS = DICT_ARG_POS_DIR + "prev2pos_Cpos_dict.txt"

''' Non-Explicit dict '''
NON_EXPLICIT_DICT_WORD_PAIRS = DICT_NON_EXPLICIT_DIR + "word_pairs.txt"
NON_EXPLICIT_DICT_PRODUCTION_RULES = DICT_NON_EXPLICIT_DIR + "production_rules.txt"
NON_EXPLICIT_DICT_DEPENDENCY_RULES = DICT_NON_EXPLICIT_DIR + "dependency_rules.txt"
NON_EXPLICIT_DICT_BROWN_CLUSTER = DICT_NON_EXPLICIT_DIR + "arg_brown_cluster.txt"

''' Implicit argument dict'''
ATTRIBUTION_NON_CONNS_DICT_PREV_CURR_CP_PRODUCTION_RULE = DICT_ATTRIBUTION_NON_CONNS_DIR + "prev_curr_CP_production_rule.txt"
ATTRIBUTION_NON_CONNS_DICT_CURR_LAST = DICT_ATTRIBUTION_NON_CONNS_DIR + "curr_last.txt"
ATTRIBUTION_NON_CONNS_DICT_LEMMA_VERBS = DICT_ATTRIBUTION_NON_CONNS_DIR + "lemma_verbs.txt"
ATTRIBUTION_NON_CONNS_DICT_2PREV_POS_LEMMA_VERB = DICT_ATTRIBUTION_NON_CONNS_DIR + "2prev_pos_lemma_verb.txt"
