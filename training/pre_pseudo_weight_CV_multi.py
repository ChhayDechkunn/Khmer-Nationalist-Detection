import pandas as pd
import pickle
import numpy as np
#for text pre-processing
import re, string
import nltk
import khmernltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
string.punctuation
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#for model_multi-building
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
#for word embedding
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
#visualzation
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import sys
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier

khmer_stopword = [' ',"មករា","កុម្ភៈ","មិនា","មេសា","ឧសភា","មិថុនា","កក្កដា","សីហា",
    "កញ្ញា","តុលា","វិច្ឆិកា","ធ្នូ","មីនា","អាទិត្យ","ច័ន្ទ","អង្គារ","ពុធ","ព្រហស្បតិ៍","សុក្រ","សៅរ៍","ចន្ទ",
                     ]
affixes = ['ការ','ភាព','សេចក្តី','អ្នក','ពួក','ទី','ខែ','ថ្ងៃ']
def remove_affixes(khmer_word, affixes):
    for affix in affixes:
        if khmer_word.startswith(affix):
            khmer_word = khmer_word[len(affix):]
        if khmer_word.endswith(affix):
            khmer_word = khmer_word[:-len(affix)]
    return khmer_word

def tokenize_kh(text, stopwords=khmer_stopword):
    # Tokenize the input text
    tokens_pos = khmernltk.pos_tag(text)
    
    # Filter out tokens with English characters, numerals, or punctuation, stopwords, and certain POS tags
    khmer_words = [
        remove_affixes(word, affixes) for word, pos in tokens_pos if (
            not re.search("[a-zA-Z0-9!@#$%^&*()_+{}\[\]:;<>,.?~\\/-]៖«»។៕!-…", word)
            and word not in stopwords
            and not any(char in word for char in '១២៣៤៥៦៧៨៩០')
            and pos not in ['o','.','1']
        )
    ]
    
    return khmer_words
    
    

def print_to_file(*args, filename='pre_pseudo_weight_CV_multi_multioutput.txt', mode='a', **kwargs):
    with open(filename, mode, encoding='utf-8') as f:
        print(*args, file=f, **kwargs)

# Redirect stdout to the file
sys.stdout = open('pre_pseudo_weight_CV_multi_multioutput.txt', 'w')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# Instantiate StratifiedKFold
skf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
with open('train_data_v5.pickle', 'rb') as f:
    X_label, y_labeled_pre = pickle.load(f)

# Load validation data
with open('val_data_v5.pickle', 'rb') as f:
   X_val, y_val_pre = pickle.load(f)

y_label=y_labeled_pre[:,1:4]

y_val=y_val_pre[:,1:4]

non_zero_rows = np.any(y_val!=0,axis=1)
y_val = y_val[non_zero_rows]
X_val = X_val[non_zero_rows]

non_zero_rows = np.any(y_label!=0,axis=1)
y_label = y_label[non_zero_rows]
X_label = X_label[non_zero_rows]

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import multilabel_confusion_matrix
    
# Custom scorer function to handle multilabel data
def custom_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
    
    
    
    
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize_kh, stop_words=khmer_stopword)),
#    ('smote',SMOTE()),
#    ('selector',SelectPercentile()),
#    ('scaler', MaxAbsScaler()),
    ('clf', MultiOutputClassifier(LogisticRegression(class_weight='balanced')))  # Classifier with balanced class weights
])

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize_kh, stop_words=khmer_stopword)),
#    ('smote',SMOTE()),
#    ('selector',SelectPercentile()),
#    ('scaler', MaxAbsScaler()),
    ('clf', MultiOutputClassifier(svm.SVC(class_weight='balanced', probability=True)))  # Classifier with balanced class weights
])

rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize_kh, stop_words=khmer_stopword)),
#    ('smote',SMOTE()),
#    ('selector',SelectPercentile()),
#    ('scaler', MaxAbsScaler()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(class_weight='balanced')))  # Classifier with balanced class weights
])

nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize_kh, stop_words=khmer_stopword)),
#    ('smote',SMOTE()),
#    ('selector',SelectPercentile()),
#    ('scaler', MaxAbsScaler()),
    ('clf', MultiOutputClassifier(ComplementNB()))  # Classifier with balanced class weights
])

xgb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize_kh, stop_words=khmer_stopword)),
#    ('smote',SMOTE()),
#    ('selector',SelectPercentile()),
    ('scaler', MaxAbsScaler()),
    ('clf', MultiOutputClassifier(XGBClassifier()))  # Classifier with balanced class weights
])

# Define hyperparameters distributions for RandomizedSearchCV for each model_multi
lr_param_dist = {
    'clf__estimator__C': [0.01,0.1, 1],
    'clf__estimator__penalty': ['l2'],
    'clf__estimator__solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'], 
}

svm_param_dist = {
    'clf__estimator__C': [0.01,0.1, 1],
    'clf__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

rf_param_dist = {
    'clf__estimator__n_estimators': [ 10, 50,100],
    'clf__estimator__max_depth': [None, 10, 20],
    'clf__estimator__min_samples_split': [2,5, 10],
    'clf__estimator__min_samples_leaf': [1,2,5]
}

nb_param_dist = {
    'clf__estimator__alpha': [0.1, 0.5, 1.0],
    'clf__estimator__fit_prior': [True, False]
}

xgb_param_dist = {
    'clf__estimator__n_estimators': [ 10, 50,100],
    'clf__estimator__max_depth': [3,5,10],
    'clf__estimator__subsample': [0.5, 0.7, 1.0],
    'clf__estimator__colsample_bytree': [0.5, 0.7, 1.0]
}

X_train = X_label
y_train = y_label
# Logistic Regression
lr_random_search = RandomizedSearchCV(lr_pipeline, lr_param_dist, cv=skf, scoring=make_scorer(custom_scorer), verbose=1, n_jobs=-1, n_iter=20)
lr_random_search.fit(X_train.flatten(), y_train)
best_lr_model_multi = lr_random_search.best_estimator_
y_pred_val_lr = best_lr_model_multi.predict(X_val.flatten())
print_to_file("Logistic Regression:")
print_to_file("Best_model_multi: ",lr_random_search.best_estimator_)
with open(f'lr_tfidf_binary_weighted_model_multi.pkl', 'wb') as file:
    pickle.dump(best_lr_model_multi, file)
print_to_file(classification_report(y_val, y_pred_val_lr))
print_to_file(multilabel_confusion_matrix(y_val, y_pred_val_lr))
print_to_file(lr_random_search.cv_results_['mean_test_score'])


X_train = X_label
y_train = y_label
nb_random_search = RandomizedSearchCV(nb_pipeline, nb_param_dist, cv=skf, scoring=make_scorer(custom_scorer), verbose=1, n_jobs=-1, n_iter=20)
nb_random_search.fit(X_train.flatten(), y_train)
best_nb_model_multi = nb_random_search.best_estimator_
y_pred_val_nb = best_nb_model_multi.predict(X_val.flatten())
print_to_file("Naive Bayes:")
print_to_file("Best_model_multi: ",nb_random_search.best_estimator_)
with open(f'nb_tfidf_binary_weighted_model_multi.pkl', 'wb') as file:
    pickle.dump(best_nb_model_multi, file)
print_to_file(classification_report(y_val, y_pred_val_nb))
print_to_file(multilabel_confusion_matrix(y_val, y_pred_val_nb))
print_to_file(nb_random_search.cv_results_['mean_test_score'])


X_train = X_label
y_train = y_label
xgb_random_search = RandomizedSearchCV(xgb_pipeline, xgb_param_dist, cv=skf, scoring=make_scorer(custom_scorer), verbose=1, n_jobs=-1, n_iter=20)
xgb_random_search.fit(X_train.flatten(), y_train)
best_xgb_model_multi = xgb_random_search.best_estimator_
y_pred_val_xgb = best_xgb_model_multi.predict(X_val.flatten())
print_to_file("XGBoost:")
print_to_file("Best_model_multi: ",xgb_random_search.best_estimator_)
with open(f'xgb_tfidf_binary_weighted_model_multi.pkl', 'wb') as file:
    pickle.dump(best_xgb_model_multi, file)
print_to_file(classification_report(y_val, y_pred_val_xgb))
print_to_file(multilabel_confusion_matrix(y_val, y_pred_val_xgb))
print_to_file(xgb_random_search.cv_results_['mean_test_score'])



X_train = X_label
y_train = y_label
rf_random_search = RandomizedSearchCV(rf_pipeline, rf_param_dist, cv=skf, scoring=make_scorer(custom_scorer), verbose=1, n_jobs=-1, n_iter=20)
rf_random_search.fit(X_train.flatten(), y_train)
best_rf_model_multi = rf_random_search.best_estimator_
y_pred_val_rf = best_rf_model_multi.predict(X_val.flatten())
print_to_file("Random Forest:")
print_to_file("Best_model_multi: ",rf_random_search.best_estimator_)
with open(f'rf_tfidf_binary_weighted_model_multi.pkl', 'wb') as file:
    pickle.dump(best_rf_model_multi, file)
print_to_file(classification_report(y_val, y_pred_val_rf))
print_to_file(multilabel_confusion_matrix(y_val, y_pred_val_rf))
print_to_file(rf_random_search.cv_results_['mean_test_score'])

X_train = X_label
y_train = y_label
svm_random_search = RandomizedSearchCV(svm_pipeline, svm_param_dist, cv=skf, scoring=make_scorer(custom_scorer), verbose=1, n_jobs=-1, n_iter=20)
svm_random_search.fit(X_train.flatten(), y_train)
best_svm_model_multi = svm_random_search.best_estimator_
y_pred_val_svm = best_svm_model_multi.predict(X_val.flatten())
print_to_file("SVM:")
print_to_file("Best_model_multi: ",svm_random_search.best_estimator_)
with open(f'svm_tfidf_binary_weighted_model_multi.pkl', 'wb') as file:
    pickle.dump(best_svm_model_multi, file)
print_to_file(classification_report(y_val, y_pred_val_svm))
print_to_file(multilabel_confusion_matrix(y_val, y_pred_val_svm))
print_to_file(svm_random_search.cv_results_['mean_test_score'])

