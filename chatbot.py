"""FAQ Chatbot - CodeAlpha AI Internship, Task 2.

Steps: collect FAQs -> preprocess with NLTK -> TF-IDF vectors ->
match user question by cosine similarity -> show best answer.
Run:  python chatbot.py
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

# 1. Collect FAQs (topic: an online store)
FAQS = [
    ("How can I track my order?",
     "Open 'My Orders' in your account and click 'Track'. A tracking link is also emailed after shipping."),
    ("What is your return policy?",
     "You can return items within 30 days of delivery if they are unused and in original packaging."),
    ("How long does delivery take?",
     "Standard delivery takes 3-5 business days. Express delivery takes 1-2 business days."),
    ("What payment methods do you accept?",
     "We accept credit/debit cards, UPI, net banking, and cash on delivery."),
    ("How do I cancel my order?",
     "Go to 'My Orders', select the order, and click 'Cancel'. Orders can be cancelled before they ship."),
    ("How do I reset my password?",
     "Click 'Forgot password' on the login page and follow the link sent to your email."),
    ("Do you offer international shipping?",
     "Yes, we ship to over 40 countries. Shipping charges and time depend on the destination."),
    ("How can I contact customer support?",
     "Email support@example.com or use the chat button. We reply within 24 hours."),
    ("How do I get a refund?",
     "Refunds are issued to the original payment method within 5-7 days after we receive the returned item."),
]

# 2. Preprocess: lowercase, tokenize, remove stopwords/punctuation, lemmatize
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop]
    return " ".join(tokens)


questions = [q for q, _ in FAQS]
answers = [a for _, a in FAQS]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform([preprocess(q) for q in questions])


# 3. Match user question with the most similar FAQ
def get_answer(user_q: str, threshold: float = 0.2) -> str:
    vec = vectorizer.transform([preprocess(user_q)])
    scores = cosine_similarity(vec, faq_vectors)[0]
    best = scores.argmax()
    if scores[best] < threshold:
        return "Sorry, I couldn't find an answer to that. Try rephrasing or contact support."
    return answers[best]


# 4. Simple chat loop
if __name__ == "__main__":
    print("FAQ Chatbot (type 'quit' to exit)")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye!")
            break
        if user:
            print("Bot:", get_answer(user))
