import streamlit as st
import nltk
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
import numpy as np
import networkx as nx

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Function to read and preprocess the text
def read_article(text):
    article = text.split(". ")  # Splits the text into sentences
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()  # Remove the last empty element
    return sentences

# Function to calculate sentence similarity
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))  # Unique words in both sentences
    vec1 = [0] * len(all_words)
    vec2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vec1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vec2[all_words.index(w)] += 1
    return 1 - cosine_distance(vec1, vec2)

# Function to generate similarity matrix
def gen_sim_matrix(sentences, stop_words):
    sim_matrix = np.zeros((len(sentences), len(sentences)))
    for i1 in range(len(sentences)):
        for i2 in range(len(sentences)):
            if i1 == i2:  # Skip if same sentence
                continue
            sim_matrix[i1][i2] = sentence_similarity(sentences[i1], sentences[i2], stop_words)
    return sim_matrix

# Function to generate summary
def gen_summary(text, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = read_article(text)
    sent_sim_mat = gen_sim_matrix(sentences, stop_words)
    sent_sim_graph = nx.from_numpy_array(sent_sim_mat)
    scores = nx.pagerank(sent_sim_graph)
    ranked_sent = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sent[i][1]))
    return ". ".join(summarize_text)

# Streamlit UI
st.title("Text Summarization App")
st.write("Upload a text file or enter text below to summarize it.")

# Text input or file upload
uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
input_text = st.text_area("Or, paste your text here:", "")

# Number of sentences in summary
num_sentences = st.slider("Number of sentences in summary:", min_value=1, max_value=10, value=3)

# Summarize button
if st.button("Summarize"):
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
    elif input_text.strip():
        text = input_text
    else:
        st.error("Please provide text input.")
        text = None

    if text:
        summary = gen_summary(text, num_sentences)
        st.subheader("Summary")
        st.write(summary)
