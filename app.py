#API Related libraries
from flask import Flask
from flask import request
from flask import jsonify

#ML/NLP related libraries
import nltk
import numpy as np
import random
import string # to process standard python strings

#from flask import request, jsonify
#app = Flask('main')
#app.config["DEBUG"] = True
app = Flask(__name__)

#get pool from corpus.txt file
f=open('corpus.txt','r',errors = 'ignore')
raw=f.read() #read file
raw=raw.lower()# converts to lowercase

int_flag=0

if(int_flag==0):
    nltk.download('punkt') # first-time use only
    nltk.download('wordnet') # first-time use only
    int_flag=1

#PRE-PROCESSING
#Tokenization of words
from nltk import word_tokenize,sent_tokenize
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words
lemmer = nltk.stem.WordNetLemmatizer()

#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#edge case: greeting inputs and responses
GREETING_INPUTS = ("hello", "hello!", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi, I am uRobo, hotel chatbot", "hey, I am uRobo, hotel chatbot", "hello :)! uRobo here!", "You are talking to uRobo!"]

#edge case: farewell case
FAREWELL_INPUTS = ("bye", "goodbye" , "thanks", "thank you")
FAREWELL_RESPONSES = ["goodbye", "have a nice day", "uRobo wishes you a nice day"]

#edge case: about questions
ABOUT_INPUTS = ("who are you?", "who is that?", "who are you", "who is that")
ABOUT_RESPONSES = ["You are talking to uRobo, hotel chatbot", "I am uRobo your hotel chatbot"]

#gets POST request& returns response
@app.route('/chatbot-main', methods=['POST']) #GET requests will be blocked
def chatbot_main():
    req_data = request.get_json()
    hotel_name = req_data['hotel_name']
    question_str = req_data['question_str']
    gen_respone=response(question_str)
    return jsonify(response=gen_respone)

#returns random greeting
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return False

#returns random farewell
def farewell(sentence):
    for word in sentence.split():
        if word.lower() in FAREWELL_INPUTS:
            return random.choice(FAREWELL_RESPONSES)
    return False

#returns random about
def about_bot(sentence):
    for word in sentence.split():
        if word.lower() in ABOUT_INPUTS:
            return random.choice(ABOUT_RESPONSES)
    return False

#vector similarity related libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#generating response from chatbot
def response(user_response):

    #check the edge cases
    greet= greeting(user_response)
    fare= farewell(user_response)
    abot= about_bot(user_response)
    
    #greetings
    if greet!=False:
        return greet

    #farewells
    elif fare!=False:
        return fare

    #analyze pool of answers file
    else:
        robo_response=''
        sent_tokens.append(user_response)
        #generate similarity vectors
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        #sort vectors, choose the best one
        idx=vals.argsort()[0][-2] # -2: 1 for input, 1 for response
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        #return response
        if(req_tfidf==0):
            robo_response=robo_response+"I am sorry! I don't understand you"
            return robo_response
        else:
            robo_response = robo_response+sent_tokens[idx]
            return robo_response


if __name__ == '__main__':
    app.run(debug=True)

