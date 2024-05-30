from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('df_model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_fraud():
    time = float(request.form.get('time'))
    amount = int(request.form.get('amount'))
    # Collect V1 to V28 data
    # v_features = [float(request.form.get(f'v{i}')) for i in range(1, 29)]

    # features = [time, amount] + v_features

    #prediction
    result = model.predict(np.array([time, 1.191857, 0.266151, 0.166480, 0.448154, 0.060018, -0.082361, 0.085110, -0.255425, 
    -0.166974, 1.612726, 1.065235, 0.489095, -0.143772, 0.635558, 0.463917, -0.114805, 
    0.570328, -0.799413, -0.141257, -0.206009, 0.502292, 0.219422, -0.408119, -0.009431, 
    0.798278, 0.137359, 0.141267, -0.206254 , amount]).reshape(1,-1))

    if result[0] == 1:
        result = 'FRAUD!'
    else:
        result = ' This Transaction Is NOT A FRAUD.'

    return result

if __name__== '__main__':
    app.run(debug=True)