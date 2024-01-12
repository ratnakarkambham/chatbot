from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import json
import random

app = Flask(__name__, static_url_path='/static')

# Load the saved data
with open('C:\\Users\\Ratnakar K\\chat bot CB\\AH BOT\\data.pickle', 'rb') as f:
    words, classes, training, output_empty = pickle.load(f)

# Load the trained model
model = load_model('C:\\Users\\Ratnakar K\\chat bot CB\\AH BOT\\chatbot_model.h5')

# Initialize NLTK and the lemmatizer
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load intents data from the JSON file
with open('C:\\Users\\Ratnakar K\\chat bot CB\\AH BOT\\intents.json') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    # Tokenize the user input
    sentence_words = nltk.word_tokenize(sentence)
    # Lemmatize and lowercase each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Tokenize the user input
    sentence_words = clean_up_sentence(sentence)
    # Create a bag of words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    "found in bag: %s" % w
    return np.array(bag)

def classify_local(sentence):
    # Generate a response from the model
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# def response(sentence):
#     results = classify_local(sentence)
#     if results:
#         while results:
#             for intent in intents['intents']:
#                 if intent['tag'] == results[0]['intent']:
#                     return random.choice(intent['responses'])
#             results.pop(0)

def response(sentence):
    try:
        results = classify_local(sentence)
        if results:
            while results:
                for intent in intents['intents']:
                    if intent['tag'] == results[0]['intent']:
                        return random.choice(intent['responses'])
                results.pop(0)
        # Return a more specific message for cases where understanding fails
        return "I apologize, but I'm not able to understand that at the moment."
    except Exception as e:
        # Handle any exception that might occur during classification
        return f"I encountered an issue: {str(e)}. I'm sorry, I couldn't understand that."







@app.route('/')
def chatbot_page():
    return render_template('he.html')

@app.route('/chatbot', methods=['POST'])
def chatbot_endpoint():
    user_input = request.form.get('user_input')
    chatbot_response = response(user_input)
    return jsonify({'response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)



