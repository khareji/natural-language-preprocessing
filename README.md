# natural-language-preprocessing
Here i have performed task related to Natural language processing , like Bag or word ,TFIDF,stemming-lemmatizer ,word2vec ,wordembeddingmodel(ANN) in stemming-lemmatizer-bow-TFIDF-word2vec-wordembeddingmodel(ANN)..ipynb 
file

furhter use navie bayes therom to find wheter mail is having spam or ham by applying bag of words i.e count vector
from sklearn.feature_extraction.text import CountVectorizer further detail is present in mailSPAMHAM.ipynb

after performing BOW in mailSPAM file , then i tried to solve fake news problem , data is donwloaded from kaggle only , there are four files related to fale news 
      
      1  fake news problem through BAG OF WORDS(fakeORreal_news_BOG_TITLE.ipynb)
      2  fake news problem through TFIDF(fakeORreal_news_TFIDF_TEXT.ipynb)
      3  fake news problem through Recurrent neural network by adding lstm (long short term memory ) layer in model (fakenewsLSTMipynb.ipynb)
      4  fake news problem through Recurrent neural network by adding lstm (long short term memory ) layer with bidirectional layer  in model (fakenewsBIDIRECTIONAL.ipynb)
     
      
# Data
kaggle data sets
# libraries
      import pandas as pd
      import re
      import nltk
      from nltk.corpus import stopwords
      from nltk.stem.porter import PorterStemmer
      from sklearn.metrics import confusion_matrix
      from sklearn.metrics import accuracy_score
      from sklearn.model_selection import train_test_split
      from sklearn.naive_bayes import MultinomialNB
      from sklearn.feature_extraction.text import CountVectorizer
      from tensorflow.keras.layers import Embedding
      from tensorflow.keras.preprocessing.sequence import pad_sequences 
      from tensorflow.keras.models import Sequential
      from tensorflow.keras.preprocessing.text import one_hot  
      from tensorflow.keras.layers import LSTM 
      from tensorflow.keras.layers import Dense 
      from tensorflow.keras.layers import Bidirectional
      
      
# Conclusion
all the model are performing well with good accuracy and detail related to each model is present in all ipynb file
      



