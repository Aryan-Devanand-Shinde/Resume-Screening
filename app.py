import nltk
import pickle
import re
import streamlit as st
import re
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# loading models rc
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

stop_words = stopwords.words('english')
special_char = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "=", "+",
                "[", "]", "{", "}", "|", "\\", ";", ":", "'", "\"", ",", "<", ">", ".",
                "/", "?", "~", "`"]

def cleaning(text):
    # removing link
    cleanedText = re.sub('http\S+\s', '', text)

    # removing mails
    cleanedText = re.sub('\S+@\S+', '', cleanedText)

    # removing hashtags
    cleanedText = re.sub('#\S+', '', cleanedText)

    # removing HTML tags
    cleanedText = re.sub(r"<.*?>", '', cleanedText)

    # removing numbers
    cleanedText = re.sub(r"\b\d+\b", '', cleanedText)

    # removing stopwords
    def rem_stop_words(cleanedText):
        clean = []
        word = cleanedText.split(' ')
        for w in word:
            if w not in stop_words:
                clean.append(w)
        cleanedText = " ".join(clean)
        return cleanedText

    cleanedText = rem_stop_words(cleanedText)

    # remove /
    cleanedText = re.sub('/s+', '', cleanedText)

    # remove special characters
    cleanedText = re.sub(r"[^a-zA-Z0-9\s]", '', cleanedText)

    # remove
    cleanedText = re.sub(r"\r\n", '', cleanedText)

    # removing extra white-space
    cleanedText = cleanedText.strip()

    return cleanedText

# web app
def main():
    st.title('Resume Screening App')
    upload_file = st.file_uploader('upload_resume',type=['txt','pdf'])

    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')

        except UnicodeDecodeError:
            resume_text = resume_bytes.decode(('latin-1'))

        cleaned_resume = cleaning(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(cleaned_resume)[0]
        st.write(prediction_id)

# main
if __name__ == "__main__":
    main()