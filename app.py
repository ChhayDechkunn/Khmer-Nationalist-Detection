import khmernltk
import re
khmer_stopword = [' ',"មករា","កុម្ភៈ","មិនា","មេសា","ឧសភា","មិថុនា","កក្កដា","សីហា",
    "កញ្ញា","តុលា","វិច្ឆិកា","ធ្នូ","មីនា","អាទិត្យ","ច័ន្ទ","អង្គារ","ពុធ","ព្រហស្បតិ៍","សុក្រ","សៅរ៍","ចន្ទ",
                     ]
affixes = ['ការ','ភាព','សេចក្តី','អ្នក','ពួក','ទី','ខែ','ថ្ងៃ']
def remove_affixes(khmer_word, affixes):
    for affix in affixes:
        if khmer_word.startswith(affix):
            khmer_word = khmer_word[len(affix):]
        if khmer_word.endswith(affix):
            khmer_word = khmer_word[:-len(affix)]
    return khmer_word

def tokenize_kh(text, stopwords=khmer_stopword):
    # Tokenize the input text
    tokens_pos = khmernltk.pos_tag(text)
    
    # Filter out tokens with English characters, numerals, or punctuation, stopwords, and certain POS tags
    khmer_words = [
        remove_affixes(word, affixes) for word, pos in tokens_pos if (
            not re.search("[a-zA-Z0-9!@#$%^&*()_+{}\[\]:;<>,.?~\\/-]៖«»។៕!-…", word)
            and word not in stopwords
            and not any(char in word for char in '១២៣៤៥៦៧៨៩០')
            and pos not in ['o','.','1']
        )
    ]
    
    return khmer_words
   
	
from flask import Flask, request, jsonify, render_template
import pickle


def remove_zero_width_spaces(text):
    return text.replace('\u200B', '')
    
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    text = request.args.get('inputText')
    binary_prediction = binary_model.predict([text])[0]
    if binary_prediction == 1:
        messages = []
        pride_prediction = pride_model.predict([text])[0]
        threat_prediction = threat_model.predict([text])[0]
        xenop_prediction = xenop_model.predict([text])[0]

        if pride_prediction == 1:
            messages.append("nationalist pride sentiment")
        if threat_prediction == 1:
            messages.append("nationalist threat sentiment")
        if xenop_prediction == 1:
            messages.append("nationalist xenophobia sentiment")

        if messages:
            message = f"The piece of text may contain {' and '.join(messages)}."
        else:
            message = "The piece of text may contain nationalist sentiment."
    return render_template('result.html', prediction=binary_prediction, message=message)

if __name__ == '__main__':
    # Load the trained model
    with open('model/lr_tfidf_binary_weighted_model.pkl', 'rb') as f:
        binary_model = pickle.load(f)
    with open('model/svm_tfidf_binary_weighted_model_pride.pkl', 'rb') as f:
        pride_model = pickle.load(f)
    with open('model/nb_tfidf_binary_weighted_model_threat.pkl', 'rb') as f:
        threat_model = pickle.load(f)
    with open('model/svm_tfidf_binary_weighted_model_anti.pkl', 'rb') as f:
        xenop_model = pickle.load(f)
  
    app.run(debug=True)