from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('sms.html')


@app.route('/Spamprediction', methods=['POST'])
def Spamprediction():
    model = pickle.load(open('Trained Model/spam-model.pkl', 'rb'))
    tfv = pickle.load(open('Trained Model/CountVectorizer-transform.pkl', 'rb'))

    if request.method == 'POST':
        message = request.form["msg"]
        data = [message]
        msg = tfv.transform(data).toarray()
        result = model.predict(msg)

    if(int(result) == 1):
        prediction = "This is a SPAM message!"
    else:
        prediction = "This is NOT a spam message."
    return(render_template("result.html", prediction_text=prediction))


if __name__ == '__main__':
    app.run(debug=True)
