# Joyce Choo 
import random
import string
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


def main():
    warnings.filterwarnings('ignore')  # to ignore any warnings

    f = open('knowledgeBase.txt', 'r', encoding='utf8', errors='ignore')
    text = f.read()  # read in knowledge base
    text = text.lower()  # lowercase text
    sentences = sent_tokenize(text)  # tokenize text
    print("Chatbot: Hi, I'm Chatbot. What is your name?")
    user_name = input()  # get user's name
    with open("data.json") as f:
        data = json.load(f)
    if user_name not in data:
        print("Chatbot: What is your email?")
        email = input()  # get user's email
        print("Chatbot: What position do you play in basketball? Enter n/a if not applicable.")
        position = input()  # get user's position
        print("Chatbot: Where did you learn how to play basketball? High school? Club? Enter n/a if not applicable.")
        level = input()  # get user's level
        print("Chatbot: Welcome", user_name, ", nice to meet you!")
    else:
        # display user's information
        print("Chatbot: Welcome back", user_name)
        print("Your email is:", data[user_name][0])
        print("Your position in basketball is:", data[user_name][1])
        print("Your level of basketball is:", data[user_name][2])
        print("In the past you liked:", data[user_name][3])
        print("In the past you disliked:", data[user_name][4])
        email = data[user_name][0]
        position = data[user_name][1]
        level = data[user_name][2]
    flag = True
    print("Chatbot: If you have any questions about basketball, please ask.")
    print("Chatbot: If you want to exit, type bye or goodbye!")
    while flag:
        user_input = input()  # get user input
        user_input = user_input.lower()  # lowercase user input
        # display a goodbye message if user wants to exit program
        if user_input == 'goodbye' or user_input == 'bye':
            print("Chatbot: Goodbye! Nice chatting with you!")
            flag = False
        # display a you're welcome message if user says thanks or thank you
        elif user_input == 'thanks' or user_input == 'thank you':
            # if user says thanks, say you're welcome instead of searching for a response
                print("Chatbot: You're welcome!")
        # display a greeting if user greets chatbot
        elif greeting(user_input):
                print("Chatbot: ", greeting(user_input))
        # display appropriate message based off user input
        else:
            print("Chatbot: ", end="")
            print(response(user_input, sentences))
            sentences.remove(user_input)
    print(user_name, " we would love to hear some feedback! Tell us something you liked about this bot.")
    user_like = input()
    print("Thank you for that feedback!")
    print("Please tell us something you disliked about this bot.")
    user_dislike = input()
    data[user_name] = [email, position, level, user_like, user_dislike]
    with open("data.json", "w+") as f:
        json.dump(data, f)


# lemmatize tokens
def lemmatizer(tokens):
    wnl = WordNetLemmatizer()
    return [wnl.lemmatize(t) for t in tokens]


# preprocess lemmas
def preprocess(text):
    remove_punct = dict((ord(punct), None) for punct in string.punctuation)
    return lemmatizer(nltk.word_tokenize(text.lower().translate(remove_punct)))


# return random greeting based off user's greeting input
def greeting(text):
    # keyword matching for greetings
    greetings = ("hello", "hi", "hey")
    responses = ["hi", "hey", "hi there", "hello"]
    text = text.lower()
    for word in text.split():
        if word in greetings:
            return random.choice(responses)


# return chatbot response based off user's input
def response(user_input, sentences):
    chatbot_response = ''
    sentences.append(user_input)
    # use preprocess as our tokenizer and english stop words
    vectorizer = TfidfVectorizer(tokenizer=preprocess, stop_words='english')
    tfidf = vectorizer.fit_transform(sentences)  # send all of our sentences into tfidf
    cos = cosine_similarity(tfidf[-1], tfidf)  # get cosine similarity values
    sorted = cos.argsort()[0][-2]  # sort cosine similarity values
    flat = cos.flatten()  # flatten sorted array
    flat.sort()  # sort flattened array
    req_tfidf = flat[-2]
    # display error message if user input does not match a keyword in knowledge base
    if req_tfidf == 0:
        chatbot_response = chatbot_response + "I'm sorry, I don't understand what you're saying."
        return chatbot_response
    # display chatbot response based off user's input from knowledge base
    else:
        chatbot_response = chatbot_response + sentences[sorted]
        return chatbot_response


if __name__ == '__main__':
    main()