import nltk
nltk.download('stopwords')
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
import numpy as np
import networkx as nx

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ") #divides the whole article into sentences
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
        # .replace("[^a-zA-Z]"," ") is used to replace any character which is not present in alphabets by spaces and .split(" ") splits the modified sentence into words based on spaces resulting a list of words
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]  
    sent2 = [w.lower() for w in sent2]  
    all_words = list(set(sent1 + sent2)) # creates a list of words occuring in sentences and set is created to ensure that words d not occur multiple times in the list
    vec1 = [0]*len(all_words)
    vec2 = [0]*len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vec1[all_words.index(w)] += 1 # for enumerating the vectors
    for w in sent2:
        if w in stopwords:
            continue
        vec2[all_words.index(w)] += 1 # for enumerating the vectors
        
    return 1 - cosine_distance(vec1, vec2) # gives us similarity between the two sentences

def gen_sim_matrix(sentences, stop_words):
    sim_matrix = np.zeros((len(sentences),len(sentences)))
    for i1 in range(len(sentences)):
        for i2 in range(len(sentences)):
            if i1 == i2:
                continue
            sim_matrix[i1][i2]=sentence_similarity(sentences[i1], sentences[i2], stop_words)
            
    return sim_matrix

def gen_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = read_article(file_name)
    sent_sim_mat = gen_sim_matrix(sentences, stop_words)
    sent_sim_graph = nx.from_numpy_array(sent_sim_mat)
    scores = nx.pagerank(sent_sim_graph)
    ranked_sent = sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sent[i][1]))
    print("\nSummary \n",". ".join(summarize_text))
    
gen_summary("nlp.txt", 3)