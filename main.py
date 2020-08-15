import random
import gensim
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gensim import utils
from gensim.models.phrases import Phraser
from gensim.test.utils import get_tmpfile
from mlxtend.feature_selection import ColumnSelector
from pandas import DataFrame
from sklearn import svm, model_selection, tree, metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
from tqdm import tqdm
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from time import time
import warnings
from timeit import default_timer as timer
from naive_bayes import BernoulliNaiveBayes


# ----------------------------------------- Print Confusion Matrix --------------------------------------------------- #
def print_conf_matrix(X, y):
    cnb = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', ComplementNB(alpha=0.345))])
    rf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=1000, max_depth=14, n_jobs=-1))])
    knn = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=99, n_jobs=-1))])
    lda = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearDiscriminantAnalysis()))])
    svc = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1))])
    lr = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial'))])
    dt = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(tree.DecisionTreeClassifier(), n_jobs=-1))])
    meta = OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1)
    meta = ComplementNB(alpha=0.36)
    # add classifiers to test synergies
    sclf = StackingClassifier(classifiers=[], meta_classifier=meta,
                              use_features_in_secondary=True, use_probas=False, verbose=1)

    # train model, get predictions, and print the test accuracy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = sclf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Accuracy: %.5f' % accuracy_score(y_pred, y_test))

    # output confusion matrix
    aux_df = ['AskReddit', 'GlobalOffensive', 'Music', 'Overwatch', 'anime', 'baseball', 'canada', 'conspiracy',
              'europe', 'funny', 'gameofthrones', 'hockey', 'leagueoflegends', 'movies', 'nba', 'nfl', 'soccer',
              'trees', 'worldnews', 'wow']
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12.8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues",
                xticklabels=aux_df,
                yticklabels=aux_df)
    plt.ylabel('Predicted')
    plt.xlabel('Actual')
    plt.title('Confusion matrix')
    plt.show()


# --------------------------------------------- Voting Ensemble ------------------------------------------------------ #
def voting_ensemble(X, y):
    cnb1 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', ComplementNB(alpha=1.353))])
    cnb2 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', ComplementNB(alpha=0.347))])
    cnb3 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', ComplementNB(alpha=0.347, special=1))])
    cnb4 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', ComplementNB(alpha=0.347, special=2))])
    svc = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1)), ])
    lr = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')) ])
    rf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=1000, max_depth=14, verbose=1, random_state=0, n_jobs=-1)), ])
    knn = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=100, n_jobs=-1)), ])

    # add base classifiers to test synergies
    model = VotingClassifier(estimators=[('cnb2', cnb2), ('svc', svc), ('lr', rf)], voting='hard', n_jobs=-1)
    scores = model_selection.cross_val_score(model, X, y, cv=4, scoring='accuracy', n_jobs=-1)
    print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std()))


# -------------------------------------------- Stacking Ensemble ----------------------------------------------------- #
def stacking_ensemble(X, y):
    cnb1 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', ComplementNB(alpha=0.347))])
    cnb2 = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ('clf', ComplementNB(alpha=0.347))])
    rf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=1000, max_depth=14, n_jobs=-1)), ])
    knn = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', KNeighborsClassifier(n_neighbors=100, n_jobs=-1)), ])
    xgb = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', XGBClassifier(objective='multi:softmax', num_class=20, n_jobs=-1)), ])
    lr = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')), ])
    lgbm = Pipeline([
        ('vect', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', LGBMClassifier(objective='multiclass', reg_lambda=1e-6, num_leaves=150,
                               n_estimators=200, learning_rate=0.07))])

    meta = OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1)
    sclf = StackingClassifier(classifiers=[cnb1, cnb2, rf, knn, xgb, lgbm], meta_classifier=meta, use_probas=True)

    # ---------------------------- 4 Fold CV ---------------------------------
    scores = model_selection.cross_val_score(sclf, X, y, cv=4, scoring='accuracy', n_jobs=-1)
    print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std()))

    # ---------------------------- GridSearch ---------------------------------
    # params = {'clf__alpha': [0.345, 0.346, 0.347]}
    # gs = GridSearchCV(cnb2, params, cv=3, scoring='accuracy', n_jobs=-1)
    # gs = gs.fit(X, y)
    # print('Best parameters: %s' % gs.best_params_)
    # print('Accuracy: %.5f' % gs.best_score_)

    # ------------------------ Output Predictions ------------------------------
    # sclf = sclf.fit(X, y)
    # y_pred = sclf.predict(X_test)
    # df_test['Category'] = y_pred
    # final_df = df_test[['id', 'Category']]
    # print(final_df.shape)
    # final_df.to_csv('./predictions.csv', encoding='utf-8', index=False)


# -------------------------------------------- Test Bernoulli NB ----------------------------------------------------- #
def test_bernoulli(X, y):
    # prepare data for our implementation of Bernoulli
    Encoder = LabelEncoder()
    Encoder.fit(y)
    y = Encoder.transform(y)
    X = CountVectorizer(binary=True).fit_transform(X)
    X = X.toarray()

    # ---------------------------- Train and Predict ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    bernoulli = BernoulliNaiveBayes()
    start = timer()
    bernoulli = bernoulli.fit(X_train, y_train)
    y_pred = bernoulli.predict(X_test)
    end = timer()
    print('Accuracy: %.5f' % accuracy_score(y_pred, y_test))
    print('Runtime: %.2f' % (end-start))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #
# ----------------------------------------------- Main Script -------------------------------------------------------- #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #

# Import Training Set
df = pd.read_csv('preprocessed_data2.csv')
df = df[pd.notnull(df['comments'])]
X = df['comments']
y = df['subreddits']

# Import Test Set (for outputting predictions)
# df_test = pd.read_csv('preprocessed_test_data2.csv')
# df_test = df_test[pd.notnull(df_test['comments'])]
# X_test = df_test['comments']

# Run Tests
print_conf_matrix(X, y)
test_bernoulli(X, y)
voting_ensemble(X, y)
stacking_ensemble(X, y)



# ------------------------------------------- Word Embeddings Test --------------------------------------------------- #
# from gensim.models import Word2Vec, Phrases
# from gensim.scripts.glove2word2vec import glove2word2vec
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# tmp_corpus = df['comments'].map(lambda x: x.split('.'))
# corpus = []
# for i in tqdm(range(len(tmp_corpus))):
#     for line in tmp_corpus[i]:
#         words = [x for x in line.split()]
#         corpus.append(words)
#
# num_of_sentences = len(corpus)
# num_of_words = 0
# for line in corpus:
#     num_of_words += len(line)
#
# print('Num of sentences - %s'%(num_of_sentences))
# print('Num of words - %s'%(num_of_words))
#
# phrases = Phrases(sentences=corpus, min_count=25, threshold=50)
# bigram = Phraser(phrases)
#
# for index,sentence in enumerate(corpus):
#     corpus[index] = bigram[sentence]
#
# # shuffle corpus
# def shuffle_corpus(sentences):
#     shuffled = list(sentences)
#     random.shuffle(shuffled)
#     return shuffled
#
# # sg - skip gram |  window = size of the window | size = vector dimension
# size = 300
# window_size = 5  # sentences weren't too long, so
# epochs = 300
# min_count = 2
# workers = 4
#
# # train word2vec model using gensim
# model = Word2Vec(corpus, window=window_size, size=size, min_count=min_count,
#                  workers=workers, iter=epochs, sample=0)
# model.save('w2v_model')
#
# model = Word2Vec.load('w2v_model')
#
# def w2v_tokenize_text(text):
#     tokens = []
#     for sent in nltk.sent_tokenize(text, language='english'):
#         for word in nltk.word_tokenize(sent, language='english'):
#             if len(word) < 1:
#                 continue
#             tokens.append(word)
#     return tokens
#
#
# from UtilWordEmbedding import MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer
#
# text_tokenized = df[['comments', 'subreddits']].apply(lambda r: w2v_tokenize_text(r['comments']), axis=1).values
#
# tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
# tfidf_vec_tr.fit(text_tokenized)
# tfidf_doc_vec = tfidf_vec_tr.transform(text_tokenized)
#
# cnb = ComplementNB()
# knn = KNeighborsClassifier(n_neighbors=20, n_jobs=-1, weights='distance')
# lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
# meta = OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1)
# sclf = StackingClassifier(classifiers=[cnb, knn, lr], meta_classifier=meta, use_probas=True)
#
# X_train_vec, X_test_vec, y_train, y_test = train_test_split(tfidf_doc_vec, y, test_size=0.2, random_state=0)
#
# sclf = KNeighborsClassifier(n_neighbors=20, n_jobs=-1, weights='distance')
# sclf = sclf.fit(X_train_vec, y_train)
# y_pred = sclf.predict(X_test_vec)
# print(accuracy_score(y_pred, y_test))
#
# aux_df = df[['subreddits']].drop_duplicates()
# conf_matrix = confusion_matrix(y_test, y_pred)
# conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize=(12.8, 6))
# sns.heatmap(conf_matrix, annot=True, cmap="Blues",
#             xticklabels=aux_df['subreddits'].values,
#             yticklabels=aux_df['subreddits'].values)
# plt.ylabel('Predicted')
# plt.xlabel('Actual')
# plt.title('Confusion matrix')
# plt.show()





# ----------------------------------------------- Doc2Vec Test ------------------------------------------------------- #
# from gensim.models.doc2vec import TaggedDocument, Doc2Vec
#
# tagged_doc = df[['comments', 'subreddits']].apply(
#     lambda r: TaggedDocument(words=w2v_tokenize_text(r['comments']), tags=[r.subreddits]), axis=1)
#
# train, test = train_test_split(df[['comments', 'subreddits']], test_size=0.2, random_state=0)
# train_tagged = train.apply(
#     lambda r: TaggedDocument(words=w2v_tokenize_text(r['comments']), tags=[r.subreddits]), axis=1)
# test_tagged = test.apply(
#     lambda r: TaggedDocument(words=w2v_tokenize_text(r['comments']), tags=[r.subreddits]), axis=1)
#
# print(train_tagged.values[30])
#
# import multiprocessing
# cores = multiprocessing.cpu_count()
# model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.00025, workers=cores)
# model_dbow.build_vocab([x for x in tqdm(tagged_doc.values)])
# model_dbow.save("d2v_model")
# model_dbow = Doc2Vec.load("d2v_model")
# from sklearn import utils
# for epoch in range(100):
#     model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]),
#                                     total_examples=len(train_tagged.values), epochs=1)
#     model_dbow.alpha -= 0.002
#     model_dbow.min_alpha = model_dbow.alpha
#
# def vec_for_learning(model, tagged_docs):
#     sents = tagged_docs.values
#     targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
#     return targets, regressors
#
# y_train, X_train = vec_for_learning(model_dbow, train_tagged)
# y_test, X_test = vec_for_learning(model_dbow, test_tagged)
#
# logreg = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial', C=1e5)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
#
# from sklearn.metrics import accuracy_score, f1_score
#
# print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
# print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
