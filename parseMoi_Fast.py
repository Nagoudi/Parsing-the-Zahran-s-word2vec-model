# encoding=utf8


from scipy import  spatial
import time
import math
from optparse import OptionParser
import numpy as np
from numpy import linalg as LA
import operator as op
import os, sys

def parse (VECTOR_PATH, WORD2VEC):
    print 'Start loading vectors ...'
    start_time = time.time();

    vecDic = {}    
    vectors = None
    fin = open(VECTOR_PATH, "rb")  
    header = fin.readline()
    vocab_size, vector_size = map(int, header.split())
    if(WORD2VEC): #for CBOW or Skip-gram
        binary_len = np.dtype(np.float32).itemsize * vector_size
    else: # for GloVe
        binary_len = np.dtype(np.float64).itemsize * vector_size

    for line_no in xrange(vocab_size):                
        word = ''
        while True:
            ch = fin.read(1)
            if ch == ' ' and WORD2VEC == 1:
                break
            elif ch == '#' and WORD2VEC==0:
				break
            word += ch    
        if(WORD2VEC):       
            vector = np.fromstring(fin.read(binary_len), np.float32)
        else:                                                       
            vector = np.fromstring(fin.read(binary_len), np.float64)                   
        word = word.strip()
        vecDic[word] = vector        
    print 'finished loading vectors ...'
    print("Number of words in the model : %s" % (len(vecDic)))
    print("Time for loading the model : %s seconds" % (time.time() - start_time))
    return vecDic



def readWordsFile(WordFilename):
    allWords=[]
    with open(WordFilename, 'r') as f:
        allWords = f.readlines()
    words = []
    for i in range(0, len(allWords)):
        if (allWords[i] != '\n'):
            words.append(allWords[i].strip())
    return words

def writeResults(allRes, outputFile):
    f = open(outputFile, 'w')
    for word, similairWords in allRes.items():
        f.write(word + '\n')
        f.write("--------------------------------------" + '\n')
        for i, e in enumerate(similairWords):
            lineFormat = "{}-->( {} , {} )"
            line = lineFormat.format(i + 1, e[0], 1 - e[1])
            f.write(line + '\n')
        f.write("--------------------------------------" + '\n')
    f.close()

def Nplus_proche(wordFilename,Dictonary,N,outputFile):

    print 'Starting looking for similair words .......'
    start_time = time.time()
    words = readWordsFile(wordFilename)

    allWords=Dictonary.keys()
    allVectors=Dictonary.values()
    norm=LA.norm(allVectors, axis=1)
    allRes={}
    index = 1
    for  word in words:
        print 'Word Number : %d'% index
        start_time_word = time.time()
        index =index+1
        if word in Dictonary:

            indexOfWord = allWords.index(word)
            vectWord =Dictonary[word]
            dots = np.dot(allVectors,vectWord)
            similarties=1-np.divide(dots,norm[indexOfWord]*norm)


            ind = np.argsort(similarties)
            ind = ind[1:(N+1)]
            result=zip([allWords[i] for i in ind], [similarties[i] for i in ind])

            allRes[word]= result

            print "-------- %s sec--------" % (time.time() - start_time_word)
        else:
            print ("word don't exist ")

    writeResults(allRes, outputFile)
    print("Time of finding similairs words : %s seconds" % (time.time() - start_time))
    print 'Done ...'
    return allRes





if __name__=='__main__':

    args = ["-m", "/home/facultemi/Desktop/Moatez files/ALL_NORM_PHRASES200_CBOW300_WIN5_NEG10_HS0_MIN10_SAMPLE","-w","/home/facultemi/Desktop/Moatez files/input.txt","-n","25","-o","/home/facultemi/Desktop/Moatez files/output.txt"]

    parser = OptionParser()

    parser.add_option("-m", "--model",
                      action="store",
                      type="string",
                      dest="modelFilename",
                      help="The name of file of Word2vec Model")
    parser.add_option("-w", "--word",
                      action="store",
                      type="string",
                      dest="WordFilename",
                      help="The name of text file that contains the used word")
    parser.add_option("-n", "--number",
                      action="store",
                      type="string",
                      dest="N",
                      help="The nnumber of similair words")
    parser.add_option("-o", "--output",
                      action="store",
                      type="string",
                      dest="outputFile",
                      help="The name of output text file")

    (options, args) = parser.parse_args(args=None, values=None)

    error = False

    if ((options.modelFilename==None) or (options.WordFilename==None) or (options.N == None) or (options.outputFile == None)):
        print "Invalid arguments use -h to view informations about arguments"
        error=True

    if(error == False):
        vecDic= parse(options.modelFilename,1)
        res = Nplus_proche(options.WordFilename,vecDic,int(options.N),options.outputFile)



    print "---------------------- Done ------------------------"

# W ='الاسلام‏'
# V=Dictonary[W]
# print np.dot(V, V)
# W = 'ديننا'
# V = Dictonary[W]
# print np.dot(V, V)

# dotsDic = dict(zip(dictionary.keys(),dots))
# vecDic = {}

# for key,vector in dictionary.iteritems():
#    vecDic[key]=1-(dotsDic[key] / (norm[key] * norm[word]))



