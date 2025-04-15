import streamlit as st
import json
import pickle
import numpy as np
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load necessary files
lemmatizer = WordNetLemmatizer()
model = load_model("chatbot_model.keras")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Load intents JSON
with open("intents.json", "r") as file:
    intents = json.load(file)

# Preprocessing function
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Bag of Words function
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Predict intent function
def predict_class(sentence):
    bow_input = np.array([bow(sentence, words)])
    prediction = model.predict(bow_input)[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(prediction) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return [{"intent": classes[r[0]], "probability": r[1]} for r in results] if results else []

# Get response function
def get_response(intents_list):
    if not intents_list:
        return "Sorry, I don't understand that. Can you rephrase?"
    
    tag = intents_list[0]['intent']
    for intent in intents["intents"]:
        if intent["intent"] == tag:  # Fix: Corrected "intent" key to "tag"
            return random.choice(intent["responses"])
    
    return "Sorry, I don't understand that. Can you rephrase?"  # Fallback response

# Streamlit UI
st.title("💬 Hannie Help")
st.write("Welcome! Type your message below and the chatbot will respond.")

# Reset Chat Button
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get chatbot response
    intents_list = predict_class(user_input)
    bot_response = get_response(intents_list)

    # Append chatbot response
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
