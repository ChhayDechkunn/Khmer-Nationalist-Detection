from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from preprocessing import tokenize_kh, remove_affixes
def remove_zero_width_spaces(text):
    return text.replace('\u200B', '')
import logging   
app = Flask(__name__)
logging.debug('tokenize_kh and remove_affixes have been imported successfully.')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
        # Load the trained models globally
    with open('model/lr_tfidf_binary_weighted_model.pkl', 'rb') as f:
        binary_model = pickle.load(f)
    with open('model/svm_tfidf_binary_weighted_model_pride.pkl', 'rb') as f:
        pride_model = pickle.load(f)
    with open('model/nb_tfidf_binary_weighted_model_threat.pkl', 'rb') as f:
        threat_model = pickle.load(f)
    with open('model/svm_tfidf_binary_weighted_model_anti.pkl', 'rb') as f:
        xenop_model = pickle.load(f)
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
    app.run(debug=True)