from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

encoder = joblib.load('ml/encoder.h5')
scaler = joblib.load('ml/en_scaler.h5')
model = joblib.load('ml/model_en.h5')


@app.route('/cars/predict', methods=['POST'])
def predict():
    car = []
    car.append(request.json['make'])
    car.append(request.json['model'])
    car.append(request.json['year'])
    car.append(int(request.json['CC']))
    car.append(int(request.json['distance']))
    car.append(request.json['transmission'])
    car.append(request.json['color'])

    car = encoder.transform([car])
    car = scaler.transform(car)
    prediction = model.predict(car)[0]
    sent = "the estimated price (EGP) is: "
    return (sent+ str(prediction) )


if __name__ == "__main__":
    app.run()