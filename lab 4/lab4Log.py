#importing libraries

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import argparse

# function for reading a new file for testing 
def  new_testfile(filename):
    newfile = pd.read_csv(filename, sep="\t", names=["docs", "class"])
    return newfile

#Breaking the read file into list
def readtestfile(filename):
    data = []
    f = open(filename, 'r')
    for line in f:
        data.append(line)

    return data

#Logistic regression with normalization
def LGnormalized(newfile):
    #Reading train files
    filename1 = "amazon_cells_labelled.txt"
    filename2 = "amazon_cells_labelled.txt"
    filename3 = "yelp_labelled.txt"
    df1 = pd.read_csv(filename1, sep="\t", names=["docs", "class"]) 
    df2 = pd.read_csv(filename2, sep="\t", names=["docs", "class"])
    df3 = pd.read_csv(filename3, sep="\t", names=["docs", "class"])
    folder = pd.concat([df1, df2, df3], axis= 0, join='inner') 

    #Normalization with stopwords
    n = set(stopwords.words("english")) 
    Vectwords =  TfidfVectorizer(use_idf= True, lowercase=True, strip_accents="ascii", stop_words=n)
    y = folder['class']
    x = Vectwords.fit_transform(folder.docs)

    #Training data given all three files
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    #Testing the given file and outputing result file
    classifier =np.array(newfile)
    classifier_vect = Vectwords.transform(classifier)
    pre = clf.predict(classifier_vect)
    score = clf.score(x_test, y_test)
    print(score)
    print(pre)
    file = open("results-lr-n.txt","a")
    for i  in pre:
        print(i)
        file.write(str(i) + "\n")
    file.close() 
    
    


def LGunnormalized(newfile):
    #reading file
    filename = "amazon_cells_labelled.txt"
    folder = pd.read_csv(filename, sep="\t", names=["docs", "class"]) 
    

    #Logistic regression without normalization
    Vectwords =  TfidfVectorizer(use_idf= False, lowercase=False, strip_accents="ascii")
    y = folder['class']
    x = Vectwords.fit_transform(folder.docs)

    #Training data
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=40)

    
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    #testing the given file using the logistic regression classifer
    classifier =np.array(newfile)
    classifier_vect = Vectwords.transform(classifier)
    pre = clf.predict(classifier_vect)
    print(pre)

    file = open("results-lr-u.txt","a")
    for i  in pre:
        print(i)
        file.write(str(i) + "\n")
    file.close() 
    
    

    #Command line argument 
def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier_type', help='classifer type- nb or lr')
    parser.add_argument('version',help='classifier version- n or u')
    parser.add_argument('file',help='the test file')
    parser = parser.parse_args()

    return parser


def main():
    args = argument()
    clasif = args.classifier_type
    ver = args.version
    fil = args.file


    if clasif == "lr":
        if ver == "n":
            data = readtestfile(fil)
            LGnormalized(data)

        else:  
            if ver == "u":
                data = readtestfile(fil)
                LGunnormalized(data)

            


if __name__=="__main__":
    main()
            