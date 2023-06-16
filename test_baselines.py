import data_helpers as dh
from BaselineModels import cnn_kim, clstm, BasicLSTM, BasicBiLSTM, BasicBiGRUs, AttentionBiLSTM
import numpy as np
import argparse

################################
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store', dest='train_corpus',
                    help='Input train file', default='data/train/TrainBinary.Shuffled.csv')
parser.add_argument('--Ar', action='store', dest='Ar', help='Choice if apply arabic normalziation', default='False')
parser.add_argument('--dev', action='store', dest='dev_corpus',
                     help='Input dev file', default='data/dev/DevBinary.csv')
parser.add_argument('--test', action='store', dest='test_corpus',
                     help='Input test file', default='data/test/TestBinary.csv')
parser.add_argument('--model_type', action='store', dest='model_type',
                     help='Baseline Model type', default='cnn')
parser.add_argument('--static', action='store', dest='STATIC',
                    help='STATIC embedding or non static for external embedding', default='True')

parser.add_argument('--rand', action='store', dest='rand',
                    help='Random Initialization for embedding or not')

parser.add_argument('--EMB_type', action='store',
                    help='embedding type, choice between skipgram CBOW or fastText', default='CBOW')

parser.add_argument('--embedding', action='store', dest='embedd_file',
                    help='Embedding Model', default="aoc_id\AOC_Skipgram.mdl")

parser.add_argument('--model_file', action='store', dest='ModelFile',
                    help='The output of the model file', default='models/CNN_Model')

args = parser.parse_args()

#====================================#
Arabic = args.Ar == 'True'
#====================================#

print('----- Load Train and Test Data --------')
X_train, Y_train, Y_train_true = dh.LoadData(args.train_corpus, ClassesDict=dh.get_classes(), Arabic=Arabic)
X_valid, Y_valid, Y_valid_true = dh.LoadData(args.dev_corpus, ClassesDict=dh.get_classes(), Arabic=Arabic)
X_test, Y_test, Y_test_true = dh.LoadData(args.test_corpus, ClassesDict=dh.get_classes(), Arabic=Arabic)

#print(X_valid)
print('---- Tokenizing Training and Testing Data ------')
X_train, X_valid, X_test, wordmap = dh.tokenizeData(X_train, X_valid, vocab_size=dh.get_vocab_size(), X_test=X_test)
X_train, X_valid, X_test = dh.paddingSequence(X_train, X_valid, maxLen=30, X_test=X_test)
n_symbols, word_map = dh.get_word_map_num_symbols(args.train_corpus)

###############################
RAND = args.rand == 'True'
###########################

Trainable = args.STATIC == "False"

##########################

if args.model_type == "cnn":
    FW = open("CNN_scores", 'w')
    CNNBaseline = cnn_kim(cnn_rand=RAND, STATIC=Trainable, ExternalEmbeddingModel=args.embedd_file,
                          EmbeddingType=args.EMB_type, n_symbols=n_symbols, wordmap=word_map)
    CNNBaseline.train_model(CNNBaseline.model, X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
    ValidScore = CNNBaseline.Evaluate_model(CNNBaseline.model, X_valid, Y_valid)
    TestScore = CNNBaseline.Evaluate_model(CNNBaseline.model, X_test, Y_test)
    FW.write('CNN Validation score: ' + str(ValidScore) + "\n")
    FW.write('CNN Test score: ' + str(TestScore) + "\n")
    FW.close()

elif args.model_type == "clstm":
    FW = open("CLSTM_scores", 'w')
    CLSTMBaseline = clstm(cnn_rand=RAND, STATIC=Trainable, ExternalEmbeddingModel=args.embedd_file,
                          EmbeddingType=args.EMB_type, n_symbols=n_symbols, wordmap=word_map)
    CLSTMBaseline.train_model(CLSTMBaseline.model, X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
    ValidScore = CLSTMBaseline.Evaluate_model(CLSTMBaseline.model, X_valid, Y_valid)
    TestScore = CLSTMBaseline.Evaluate_model(CLSTMBaseline.model, X_test, Y_test)
    FW.write('CLSTM Validation score: ' + str(ValidScore) + "\n")
    FW.write('CLSTM Test score: ' + str(TestScore) + "\n")
    FW.close()

elif args.model_type == "bilstm":
    FW = open("BiLSTM_scores", 'w')
    BiLSTMBaseline = BasicBiLSTM(cnn_rand=RAND, STATIC=Trainable, ExternalEmbeddingModel=args.embedd_file,
                                 EmbeddingType=args.EMB_type, n_symbols=n_symbols, wordmap=word_map)
    BiLSTMBaseline.train_model(BiLSTMBaseline.model, X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
    ValidScore = BiLSTMBaseline.Evaluate_model(BiLSTMBaseline.model, X_valid, Y_valid)
    TestScore = BiLSTMBaseline.Evaluate_model(BiLSTMBaseline.model, X_test, Y_test)
    FW.write('BiLSTM Validation score: ' + str(ValidScore) + "\n")
    FW.write('BiLSTM Test score: ' + str(TestScore) + "\n")
    FW.close()

elif args.model_type == "lstm":
    FW = open("LSTM_scores", 'w')
    LSTMBaseline = BasicLSTM(cnn_rand=RAND, STATIC=Trainable, ExternalEmbeddingModel=args.embedd_file,
                             EmbeddingType=args.EMB_type, n_symbols=n_symbols, wordmap=word_map)
    LSTMBaseline.train_model(LSTMBaseline.model, X_train, Y_train=Y_train, X_valid=X_valid, Y_valid=Y_valid)
    ValidScore = LSTMBaseline.Evaluate_model(LSTMBaseline.model, X_valid, Y_valid)
    TestScore = LSTMBaseline.Evaluate_model(LSTMBaseline.model, X_test, Y_test)
    FW.write('LSTM Validation score: ' + str(ValidScore) + "\n")
    FW.write('LSTM Test score: ' + str(TestScore) + "\n")
    FW.close()

elif args.model_type == "attention_bilstm":
    FW = open("AttentionBiLSTM_scores", 'w')
    AttentionBiLSTMBaseline = AttentionBiLSTM(cnn_rand=RAND, STATIC=Trainable,
                                              ExternalEmbeddingModel=args.embedd_file, EmbeddingType=args.EMB_type,
                                              n_symbols=n_symbols, wordmap=word_map)
    AttentionBiLSTMBaseline.train_model(AttentionBiLSTMBaseline.model, X_train, Y_train=Y_train, X_valid=X_valid,
                                        Y_valid=Y_valid)
    ValidScore = AttentionBiLSTMBaseline.Evaluate_model(AttentionBiLSTMBaseline.model, X_valid, Y_valid)
    TestScore = AttentionBiLSTMBaseline.Evaluate_model(AttentionBiLSTMBaseline.model, X_test, Y_test)
    FW.write('Attention BiLSTM Validation score: ' + str(ValidScore) + "\n")
    FW.write('Attention BiLSTM Test score: ' + str(TestScore) + "\n")
    FW.close()
