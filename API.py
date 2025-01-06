import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  

# WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load data set and model
data_file = open("Bot_Trainning/Train_Bot.json").read()
intents = json.loads(data_file)
model = load_model('Bot_Trainning/chatbot.h5')
words = pickle.load(open('Bot_Trainning/words.pkl', 'rb'))
classes = pickle.load(open('Bot_Trainning/classes.pkl', 'rb'))

# get chatbot response
def chatbot_response(user_message):
    # Tokenize and lemmatize
    user_message = nltk.word_tokenize(user_message)
    user_message = [lemmatizer.lemmatize(word.lower()) for word in user_message]

    # Create a bag of words
    bag = [0] * len(words)
    for word in user_message:
        if word in words:
            bag[words.index(word)] = 1
    bag = np.array(bag).reshape(1, -1)

    # the model to predict  
    result = model.predict(bag)
    predicted_class = classes[np.argmax(result)]
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            responses = intent['responses']
            return random.choice(responses)

# Define the route
@app.route('/chat', methods=['GET'])
def chat():
    return jsonify({"message": "Welcome to the Chatbot API!"})

# Define the route to get a response
@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json['user_message']
    bot_response = chatbot_response(user_message)
    return jsonify({'bot_response': bot_response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
