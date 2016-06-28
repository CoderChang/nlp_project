CWD = '/home/slhome/cc001/Documents/nlp/nlp_project/'

# directories:
DATA_DIR = CWD + 'data/'

TRAIN_DIR = CWD + 'train_dataset/'
TEST_DIR = CWD + 'test_dataset/'
MODEL_TRAINER_DIR = CWD + 'model_trainer/'
DICT_DIR = CWD + 'dict/'

TRAIN_OUT_FEATURE_DIR = CWD + 'train_output/feature/'
TRAIN_OUT_MODEL_DIR = CWD + 'train_output/model/'

TEST_OUT_FEATURE_OUT_DIR = CWD + 'test_output/feature_output/'
TEST_OUT_MODEL_OUT_DIR = CWD + 'test_output/model_output/'

DICT_CONN_DIR = DICT_DIR + 'connective/'
DICT_ARG_POS_DIR = DICT_DIR + 'argument_position/'
DICT_NON_EXPLICIT_DIR = DICT_PATH + "non_explicit_classifier/"

# file paths:
TRAIN_DATA_PATH = TRAIN_DIR + 'pdtb-data.json'
TRAIN_PARSES_PATH = TRAIN_DIR + 'pdtb-parses.json'

TRAIN_FEATURE_CONN_CL = TRAIN_OUT_FEATURE_DIR + 'conn_cl_feature.txt'

TRAIN_MODEL_CONN_CL = TRAIN_OUT_MODEL_DIR + 'conn_cl.pickle'
TRAIN_MODEL_ARG_POS_CL = TRAIN_OUT_MODEL_DIR + 'arg_pos_cl.pickle'


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

''' Non-Explicit Dict '''
NON_EXPLICIT_DICT_WORD_PAIRS = DICT_NON_EXPLICIT_DIR + "word_pairs.txt"
NON_EXPLICIT_DICT_PRODUCTION_RULES = DICT_NON_EXPLICIT_DIR + "production_rules.txt"
NON_EXPLICIT_DICT_DEPENDENCY_RULES = DICT_NON_EXPLICIT_DIR + "dependency_rules.txt"
NON_EXPLICIT_DICT_BROWN_CLUSTER = DICT_NON_EXPLICIT_DIR + "arg_brown_cluster.txt"
