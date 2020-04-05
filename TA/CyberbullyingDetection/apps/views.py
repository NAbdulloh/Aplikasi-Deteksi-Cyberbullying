from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    return render(request,'home.html')

import re
import string
import unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import xlsxwriter
import os
import numpy as np

import nltk
nltk.download('punkt')

def remove_stopword(str):
	stop_words = set(stopwords.words('E:/Kuliah/TUGAS AKHIR/Apps/TA/CyberbullyingDetection/stopwordID.csv'))
	word_tokens = word_tokenize(str)
	filtered_sentence = [w for w in word_tokens if not w in stop_words]

	return ' '.join(filtered_sentence)
	    
def remove_sentence(str):
    word = str.split()
    wordCount = len(word)
    if(wordCount <= 1):
       str = ''

    return str

def remove_url(str):
	str = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', str)
	return str
	    
def remove_digit(str):
	return re.sub(r'[^a-z ]*([.0-9])*\d', '', str)

def remove_non_ascii(str):
	str = unicodedata.normalize('NFKD', str).encode('ascii', 'ignore').decode('utf-8', 'ignore')
	return str

def remove_twitter_char(str):  
	#     # html
	#     str = re.sub(r'<[^>]+>', ' ', str) 
	    # mention
	str = re.sub(r'(?:@[\w_]+)', ' ', str)
	    # hashtag
	str = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", " ", str)
	    # RT/cc
	str = str = re.sub('RT', '', str)

	return str

def remove_punctuation(str):
	return re.sub(r'[^\s\w]', ' ', str)

def remove_add_space(str):
	str = re.sub('[\s]+', ' ', str)
	return str

def casefolding(str):  
	str = str.lower()   
	return str

def remove_repeated_character(str):
	str = re.sub(r'(.)\1{2,}', r'\1', str)
	return str

def normalize_slang_word(str):
	text_list = str.split(' ')
	slang_words_raw = pd.read_csv('E:/Kuliah/TUGAS AKHIR/Apps/TA/CyberbullyingDetection/slang_word_list.csv', sep=',', header=None)
	slang_word_dict = {}

	for item in slang_words_raw.values:
	    slang_word_dict[item[0]] = item[1]

	    for index in range(len(text_list)):
	        if text_list[index] in slang_word_dict.keys():
	            text_list[index] = slang_word_dict[text_list[index]]

	return ' '.join(text_list)

def stemm(str):
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	text = stemmer.stem(str)
	    
	return text

def preprocessing(str):
	    #str = remove_url(str)
	str = remove_twitter_char(str)
	str = remove_digit(str)
	str = remove_non_ascii(str)
	str = casefolding(str)
	str = remove_punctuation(str)
	str = remove_repeated_character(str)
	str = normalize_slang_word(str)
	str = stemm(str)
	#     #str = remove_sentence(str)
	str = remove_add_space(str)
	str = remove_stopword(str)
	    
	return str

def hasil(request):

	import pandas as pd
	data = dataset = pd.read_excel("cleandata.xlsx")

	X = data['Tweet']
	y = data['Label']

	from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
	from sklearn.pipeline import Pipeline
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.linear_model import LogisticRegression
	from sklearn.svm import LinearSVC
	from sklearn.naive_bayes import BernoulliNB
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
	import nltk


	teks = request.GET['teks']

	#Multinomial Naive Bayes
	pipeline_mnb = Pipeline([
	                          ('vect', CountVectorizer()),
	                          ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
	                          ('clf', MultinomialNB(alpha=1))
	])
	X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state = 0)

	pipeline_mnb.fit(X_train, y_train)
	predictions_mnb = pipeline_mnb.predict(X_test)
	
	preprposes = preprocessing(teks)
	kalimat = [preprposes]

	tes_log = pipeline_mnb.predict_proba(kalimat)
	arr = np.array(tes_log)
	ind_pos = [0]

	a = arr[0]

	bCyber = a[0]
	CyberBull = a[1]
	positif = round(bCyber,2)
	negatif = round(CyberBull,2)
	hasilPositif = positif
	hasilNegatif = negatif

	# tes_log = pipeline_mnb.predict(kalimat)
	# if (tes_log == 1):
	# 	hasil = "Cyberbullying"
	# else:
	# 	hasil = "Bukan Cyberbullying"


	#print(tes_log)

	res = accuracy_score(y_test, predictions_mnb)

	return render(request, 'home.html', {'bully':hasilNegatif, 'bukan':hasilPositif})