import matplotlib.pyplot as plt
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
text = ""
with open(pdf_path, "rb") as pdf_file:
pdf_reader = PyPDF2.PdfReader(pdf_file)
for page in pdf_reader.pages:
text += page.extract_text()
return text
# Sample chatbot response and the path to the PDF file
chatbot_response = "Mental health is an important aspect of overall well-being. Schools
can play a crucial role in promoting mental health and well-being of students 1 . Children
may have trouble communicating with others both at school and at home, which may lead
to poor self-esteem, poor academic and social success, and a high dropout rate 2 .
Separation anxiety is a common issue among children, which can affect their daily activities 
and tasks like going to school or peer interaction. It is characterized by experiencing extreme 
anxiety or even having panic attacks and completely hampers the functionality of a child 2 
. Attachment may be understood as a bond between children and their parents or caregivers 
that affects the child's growth and their ability to build meaningful relationships in life.
Caregivers or parents may notice that a child has problemswith emotional attachment as 
early as their first year of birth. However, with care and patience, it is possible to overcome
attachment challenges ."
pdf_path = "minor/CBSE_MH_Manual.pdf"
# Extract text from the PDF
pdf_content = extract_text_from_pdf(pdf_path)
# Preprocess the text (remove punctuation, convert to lowercase, etc.)
def preprocess_text(text):
text = text.lower()
text = re.sub(r'[^\w\s]', '', text)
return text
chatbot_response = preprocess_text(chatbot_response)
pdf_content = preprocess_text(pdf_content)
27
# Calculate cosine similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([chatbot_response, pdf_content])
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
# Visualize the similarity
plt.figure(figsize=(8, 4))
plt.bar(["Chatbot Response", "PDF Content"], [cosine_sim, 1.0])
plt.title("Cosine Similarity")
plt.ylabel("Similarity")
plt.ylim(0, 1.2)
plt.show()
