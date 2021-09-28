import nltk
from nltk.tokenize import word_tokenize
from math import log2
from numpy import argsort, zeros

ps = nltk.stem.PorterStemmer()

# Input query
query_input = input("Query: ")
query_terms = [ps.stem(term) for term in word_tokenize(query_input)]

# Analyze data for query
## Load data
file = open('animals.txt'); 
lines = file.read().splitlines(); 
file.close()

# Make two lists (title and text) which contain the title (name of animal) and text (description of animal) 
# for each entry in the text file
titles = lines[0::4]
texts = lines[2::4]

## Tokenization
tokenized_texts = []

for text in texts:
    tokenized_texts.append([ps.stem(_) for _ in word_tokenize(text)])

amount_of_docs = len(texts)
sum_of_doc_len = 0

for doc in tokenized_texts:
    sum_of_doc_len += len(doc)

average_len_of_docs = sum_of_doc_len/len(texts);

b = 0.75
k1 = 1.2

tf_idf_document_scores = zeros(amount_of_docs)
bm25_document_scores = zeros(amount_of_docs)

for i, term in enumerate(query_terms):
    ## Count occurences
    document_occurences = [text.count(term) for text in tokenized_texts]

    ## Calculate document score
    amount_of_docs_with_term = [document_occurence > 0 for document_occurence in document_occurences].count(True)

    for i in range(0, len(document_occurences)):
        occurrences = document_occurences[i]
        terms_in_doc = len(tokenized_texts[i])
        
        tf_idf_document_scores[i] += ((occurrences/terms_in_doc)*log2(amount_of_docs/amount_of_docs_with_term))
        bm25_document_scores[i] += (((occurrences*(k1+1))/(terms_in_doc+(k1*(1-b+b*(terms_in_doc/average_len_of_docs))))) \
            *log2((amount_of_docs-amount_of_docs_with_term+0.5)/(amount_of_docs_with_term+0.5)))
    
# Output top 5 best matches + TF/IDF
top_list = []
amount_of_elements_wanted = 5
index_of_best_matches = argsort(bm25_document_scores)[-amount_of_elements_wanted:].tolist()
index_of_best_matches.reverse()

print("{} best matches using Okapi BM25:".format(amount_of_elements_wanted))
for i, match_idx in enumerate(index_of_best_matches):
    print("{}. {} | TF/IDF:{} BM25: {}".format(i+1,titles[match_idx], tf_idf_document_scores[match_idx], bm25_document_scores[match_idx]))