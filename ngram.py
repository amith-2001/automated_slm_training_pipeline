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

def train_ngram_model(tokens, n=3):
    n_grams = list(ngrams(tokens, n, pad_right=True, pad_left=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
    n_gram_freq = Counter(n_grams)
    context_counts = defaultdict(int)
    for gram in n_grams:
        context_counts[gram[:-1]] += n_gram_freq[gram]
    n_gram_prob = {gram: freq / context_counts[gram[:-1]] for gram, freq in n_gram_freq.items() if context_counts[gram[:-1]] != 0}
    return n_gram_prob

def predict_next_words(model, initial_words, num_predictions=2):
    if len(initial_words) != 2:
        raise ValueError("Initial words must be a tuple of exactly 2 words.")
    current_context = tuple(initial_words)
    predicted_words = []
    for _ in range(num_predictions):
        next_words = {gram[-1]: prob for gram, prob in model.items() if gram[:-1] == current_context}
        if not next_words:
            break
        next_word = max(next_words, key=next_words.get)
        predicted_words.append(next_word)
        current_context = (current_context[1], next_word)
    return predicted_words

# Longer example text
text = """Alice was beginning to get very tired of sitting by her sister on the bank, 
and of having nothing to do. Once or twice she had peeped into the book her sister was reading, 
but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 
'without pictures or conversation?' So she was considering in her own mind (as well as she could, 
for the hot day made her feel very sleepy and stupid) whether the pleasure of making a daisy chain 
would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with 
pink eyes ran close by her."""

tokens = preprocess_text(text)
print("Tokens:", tokens)

model = train_ngram_model(tokens)

print("Trained n-gram contexts and probabilities:")
for context, prob in model.items():
    print(f"{context}: {prob}")
initial_words = ("sitting", "sister")  # Example initial words
predictions = predict_next_words(model, initial_words)
print("Next words:", predictions)
print(len(predictions))
