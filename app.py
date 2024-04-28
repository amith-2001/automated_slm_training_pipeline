import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import string
from collections import defaultdict, Counter

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define a function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return [token for token in tokens if token.strip()]

# Define a function to train the n-gram model
def train_ngram_model(tokens, n=3):
    n_grams = list(ngrams(tokens, n, pad_right=True, pad_left=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    n_gram_freq = Counter(n_grams)
    context_counts = defaultdict(int)
    for gram in n_grams:
        context_counts[gram[:-1]] += n_gram_freq[gram]
    n_gram_prob = {gram: freq / context_counts[gram[:-1]] for gram, freq in n_gram_freq.items() if context_counts[gram[:-1]] != 0}
    return n_gram_prob

# Define a function to predict the next words
def predict_next_words(model, initial_words, num_predictions=1):
    if len(initial_words) != 2:
        raise ValueError("Initial words must be a tuple of exactly 2 words.")
    current_context = tuple(initial_words)
    full_sentence = list(initial_words)
    for _ in range(num_predictions):
        next_words = {gram[-1]: prob for gram, prob in model.items() if gram[:-1] == current_context}
        if not next_words:
            break
        next_word = max(next_words, key=next_words.get)
        full_sentence.append(next_word)
        current_context = (current_context[1], next_word)
    return ' '.join(full_sentence)

# Streamlit application layout
st.title('N-Gram Model Text Predictor')

# Text input for user
user_input = st.text_area("Enter a large body of text to train the model:")

# Text input for initial words to predict the sequence
initial_input = st.text_input("Enter 2 words to start the prediction (separated by spaces):")

# Slider for choosing the number of predictions
num_predictions = st.slider("Select the number of words to predict:", min_value=1, max_value=10, value=5, help="Selecting fewer words generally results in more accurate predictions.")

# Button to trigger prediction
if st.button('Generate Text'):
    if user_input and initial_input:
        tokens = preprocess_text(user_input)
        model = train_ngram_model(tokens)
        initial_words = tuple(initial_input.split())
        if len(initial_words) == 2:
            sentence = predict_next_words(model, initial_words, num_predictions)
            st.write("Complete sentence:", sentence)
        else:
            st.error("Please enter exactly 2 words to start the prediction.")
    else:
        st.error("Please enter the required text and initial words.")
