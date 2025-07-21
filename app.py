from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# trained model को load करो
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # form से input values लो
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    result = "Risk of Heart Failure!" if output == 1 else "No Risk."
    return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
