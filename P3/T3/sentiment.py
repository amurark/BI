import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import numpy as np
import collections
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):#completed
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):#For each element in a list, a tuple is produced with (counter, element).
            #Stripts the sentence of trailing or leading spaces, splits it into words using spaces, then only takes words which have a length of greater than 3.
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)
    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):#completed
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa

    train_pos_wo_stopwords = []
    train_neg_wo_stopwords = []
    for line in train_pos:
        line_wo_stopwords = list(set(line) - stopwords)
        train_pos_wo_stopwords.append(line_wo_stopwords)
    train_pos_wo_stopwords = np.asarray(train_pos_wo_stopwords)
    #np.save("train_pos_wo_stopwords", train_pos_wo_stopwords)

    for line in train_neg:
        line_wo_stopwords = list(set(line) - stopwords)
        train_neg_wo_stopwords.append(line_wo_stopwords)
    train_neg_wo_stopwords = np.asarray(train_neg_wo_stopwords)
    #np.save("train_neg_wo_stopwords", train_neg_wo_stopwords)

    #train_pos_wo_stopwords = np.load("train_pos_wo_stopwords.npy")
    #train_neg_wo_stopwords = np.load("train_neg_wo_stopwords.npy")
    single_list_pos = np.hstack(train_pos_wo_stopwords)
    single_list_neg = np.hstack(train_neg_wo_stopwords)

    #Create dictionary for positive words
    unique, counts = np.unique(single_list_pos, return_counts=True)
    dt = dict(zip(unique, counts))

    #Create dictionary for negative words
    unique1, counts1 = np.unique(single_list_neg, return_counts=True)
    dt1 = dict(zip(unique1, counts1))

    for key,value in dt.items():
        if value < int(np.shape(train_pos_wo_stopwords)[0])/100:
            del(dt[key])

    for key1,value1 in dt1.items():
        if value1 < (np.shape(train_neg_wo_stopwords)[0])/100:
            del(dt1[key1])

    features = []
    for key,value in dt.items():
        if dt1.has_key(key):
            if value > dt1[key] and value >= 2*dt1[key]:
                #delete value from dt1
                del(dt1[key])
            elif value < dt1[key] and 2*value <= dt1[key]:
                #delete value from dt
                del(dt[key])
            else:
                del(dt1[key])
                del(dt[key])


    features.extend(list(dt.keys()))
    features.extend(list(dt1.keys()))
    #np.save("features", np.asarray(features))

    #features = list(np.load("features.npy"))

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for text in train_pos:
        bin_list = [0] * len(features)
        for idx, val in enumerate(features):
            if val in text:
                bin_list[idx] = 1

        train_pos_vec.append(bin_list)

    #np.save("train_pos_vec", np.asarray(train_pos_vec))

    for text in train_neg:
        bin_list = [0] * len(features)
        for idx, val in enumerate(features):
            if val in text:
                bin_list[idx] = 1

        train_neg_vec.append(bin_list)

    #np.save("train_neg_vec", np.asarray(train_neg_vec))

    for text in test_pos:
        bin_list = [0] * len(features)
        for idx, val in enumerate(features):
            if val in text:
                bin_list[idx] = 1

        test_pos_vec.append(bin_list)

    #np.save("test_pos_vec", np.asarray(test_pos_vec))

    for text in test_neg:
        bin_list = [0] * len(features)
        for idx, val in enumerate(features):
            if val in text:
                bin_list[idx] = 1

        test_neg_vec.append(bin_list)

    #np.save("test_neg_vec", np.asarray(test_neg_vec))

    # train_pos_vec = list(np.load("train_pos_vec.npy"))
    # train_neg_vec = list(np.load("train_neg_vec.npy"))
    # test_pos_vec = list(np.load("test_pos_vec.npy"))
    # test_neg_vec = list(np.load("test_neg_vec.npy"))

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec











def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """

    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []
    for i, text in enumerate(train_pos):
        labeled_train_pos.append(LabeledSentence(words=text, tags=['train_pos_%s' % i]))

    for i, text in enumerate(train_neg):
        labeled_train_neg.append(LabeledSentence(words=text, tags=['train_neg_%s' % i]))

    for i, text in enumerate(test_pos):
        labeled_test_pos.append(LabeledSentence(words=text, tags=['test_pos_%s' % i]))

    for i, text in enumerate(test_neg):
        labeled_test_neg.append(LabeledSentence(words=text, tags=['test_neg_%s' % i]))

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []
    # Use the docvecs function to extract the feature vectors for the training and test data
    for i, text in enumerate(train_pos):
        train_pos_vec.append(model.docvecs['train_pos_%s' % i])

    for i, text in enumerate(train_neg):
        train_neg_vec.append(model.docvecs['train_neg_%s' % i])

    for i, text in enumerate(test_pos):
        test_pos_vec.append(model.docvecs['test_pos_%s' % i])

    for i, text in enumerate(test_neg):
        test_neg_vec.append(model.docvecs['test_neg_%s' % i])

    # np.save("train_pos_vec_dtv", np.asarray(train_pos_vec))
    # np.save("train_neg_vec_dtv", np.asarray(train_neg_vec))
    # np.save("test_pos_vec_dtv", np.asarray(test_pos_vec))
    # np.save("test_neg_vec_dtv", np.asarray(test_neg_vec))
    #
    # train_pos_vec = list(np.load("train_pos_vec_dtv.npy"))
    # train_neg_vec = list(np.load("train_neg_vec_dtv.npy"))
    # test_pos_vec = list(np.load("test_pos_vec_dtv.npy"))
    # test_neg_vec = list(np.load("test_neg_vec_dtv.npy"))

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec












def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    X = train_pos_vec + train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    from sklearn.naive_bayes import BernoulliNB
    clf1 = BernoulliNB()
    nb_model = clf1.fit(X, Y)

    from sklearn.linear_model import LogisticRegression
    clf2 = LogisticRegression()
    lr_model = clf2.fit(X, Y)
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LogisticRegression Model that are fit to the training data.
    """
    X = train_pos_vec + train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    from sklearn.naive_bayes import GaussianNB
    clf1 = GaussianNB()
    nb_model = clf1.fit(X, Y)

    from sklearn.linear_model import LogisticRegression
    clf2 = LogisticRegression()
    lr_model = clf2.fit(X, Y)

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):#completed
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    X = test_pos_vec + test_neg_vec
    Y = ["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)
    pred_Y = model.predict(X)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(Y, pred_Y)
    print(cm)
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]
    tn = cm[0][0]
    accuracy = float(tp + tn)/float(len(Y))
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
