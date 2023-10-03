"""
Created on Fri May 8 17:57:49 2020

@author: Sai Venkata Aditya Arepalli
"""
#CS244:Data Science Project - Clustering:Unsupervised Learning

#Model() function is for computing the vector for compound words which are made up of two or more simple words
#This function splits compound words into simple words, computes vectors of simple words, adds them to get vector for compound word
#This function is necessary as Word2Vec can give vectors only for simple words
def Model(vector):
    Z=[]
    Z=vector.split('_')
    for i in Z:
        W=[]
        W.append(np.array(model[i]))
    List=np.array(W)
    return np.sum(List,0)


#cosine_similarity() function takes two words as arguments.
#It returns cosine of the angle between the vectors of the two imput words
#As the value returned increases, then the similarity between words increases.
def cosine_similarity(element,reference):
    try:
        if (('_' not in element) and ('_' not in reference)):
            similarity=np.dot(model[element],model[reference])/(np.linalg.norm(model[element])*np.linalg.norm(model[reference]))
            return similarity
        elif (('_' in element) and ('_' not in reference)):
            element_vec=Model(element)
            similarity=np.dot(element_vec,model[reference])/(np.linalg.norm(element_vec)*np.linalg.norm(model[reference]))
            return similarity
        elif (('_' not in element) and ('_' in reference)):
            reference_vec=Model(reference)
            similarity=np.dot(model[element],reference_vec)/(np.linalg.norm(model[element])*np.linalg.norm(reference_vec))
            return similarity
        else:
            reference_vec=Model(reference)
            element_vec=Model(element)
            similarity=np.dot(element_vec,reference_vec)/(np.linalg.norm(element_vec)*np.linalg.norm(reference_vec))
            return similarity
    except KeyError:
        return 'not in vocabulary'


#Program execution begins from here
#Importing libraries
import numpy as np
import pandas as pd
import gensim


#Data preprocessing
#Reading data and storing word id and word list into two different variables 
dataset=pd.read_csv('catname.txt',sep=",",header=None)
word_id=dataset.iloc[:,0].values
words=dataset[1].str.lower()
#Splitting compound words into simple words, removing special characters and joining them back by using '_'.
words=words.apply(lambda x:x.split())
for element in words:
    if '&' in element:
        element.remove('&')
    elif '/' in element:
        element.remove('/')
    elif '-' in element:
        element.remove('-')
for element in range(len(words)):
       words=[["_".join(i)] for i in words]
#In the above step, we get a list of list of strings.
#Converting into list of strings named word
word=[]
for element in words:
    str1=''.join(element)
    word.append(str1)
#Storing name of categories in two different variables: one without processing and other with processing
categories_before_processing=['Arts & Entertainment','College & University', 'Food','Professional & other places','Nightlife Spot','Recreation & Outdoors','Shop & Service','Travel & Transport','Residence']
categories_after_processing=['arts_entertainment','college_university','food','professional_other_places','nightlife_spot','recreation_outdoors','shop_service','travel_transport','residence']


#Building the model. Pre-trained google news vectors file has to be downloaded
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


#Finding the similarity between words of dataset and categories
#Here a word is taken from input dataset,cosine similarity is calculated with all words in categories
#After similarity with all words of categores is obtained,the word from dataset is classified into category with which it has highest similarity
clusters=[]
for element in word:
    distances=[]
    for reference in categories_after_processing:
        temp=cosine_similarity(element,reference)
        distances.append(temp)
    if 'not in vocabulary' in distances:
        clusters.append('Category unknown')
    else:
        max_index=distances.index(max(distances))
        clusters.append(categories_before_processing[max_index])


#Putting the result into a csv file
#Storing the list of words of input dataset without pre-processing
words_list=[]
for element in dataset[1]:
    words_list.append(str(element))
#Creating a dictionary to store the result
categories_list={'Word-Id':word_id,'Word':words_list,'Category':clusters}
#Creating dataframe from the above dictionary and converting into csv file
df=pd.DataFrame(categories_list,columns=['Word-Id','Word','Category'])
df.to_csv('Project_Output.csv',index=False,header=True)