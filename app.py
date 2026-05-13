import streamlit as st
import nltk

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer
nltk.download('punkt')

# Page setup
st.set_page_config(page_title="Food FAQ Chatbot", page_icon="🍔")

st.title("🍔 AI Food FAQ Chatbot")

st.write("Ask food-related questions and get smart answers.")

# FAQ Questions and Answers
faq_questions = [
    "what is biryani",
    "what is pizza",
    "what is burger",
    "what is pasta",
    "what is dosa",
    "what is ice cream",
    "what is fried rice",
    "what is sandwich"
]

faq_answers = [
    "Biryani is a popular rice dish made with spices and meat or vegetables.",

    "Pizza is an Italian dish made with bread base, cheese, and toppings.",

    "Burger is a sandwich made with buns and a vegetable or meat patty.",

    "Pasta is a traditional Italian food made from wheat dough.",

    "Dosa is a South Indian crispy pancake made from rice batter.",

    "Ice cream is a frozen sweet dessert available in many flavors.",

    "Fried rice is cooked rice stir-fried with vegetables, eggs, or meat.",

    "Sandwich is made by placing vegetables, cheese, or meat between bread slices."
]

# User input
user_question = st.text_input("Enter your question:")

# Button
if st.button("Get Answer"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")

    else:

        # Add user question to list
        all_questions = faq_questions + [user_question]

        # Convert text to vectors
        vectorizer = CountVectorizer().fit_transform(all_questions)

        # Similarity checking
        similarity = cosine_similarity(vectorizer[-1], vectorizer[:-1])

        # Best match index
        index = similarity.argmax()

        # Similarity score
        score = similarity[0][index]

        # Threshold check
        if score > 0.2:
            st.success(faq_answers[index])
        else:
            st.error("Sorry, I don't know the answer.")