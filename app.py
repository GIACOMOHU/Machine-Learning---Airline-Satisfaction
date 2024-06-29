# if you encounter problems a possible solution could be this:
# streamlit==1.31.1
# scikit-learn==1.3.2

import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the Random Forest model
try:
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        gender = int(request.form['Gender'])
        customer_type = int(request.form['CustomerType'])
        age = float(request.form['Age'])
        type_of_travel = int(request.form['TypeOfTravel'])
        flight_class = int(request.form['Class'])
        flight_distance = float(request.form['FlightDistance'])
        inflight_wifi_service = int(request.form['InflightWifiService'])
        departure_arrival_time_convenient = int(request.form['DepartureArrivalTimeConvenient'])
        ease_of_online_booking = int(request.form['EaseOfOnlineBooking'])
        gate_location = int(request.form['GateLocation'])
        food_and_drink = int(request.form['FoodAndDrink'])
        online_boarding = int(request.form['OnlineBoarding'])
        seat_comfort = int(request.form['SeatComfort'])
        inflight_entertainment = int(request.form['InflightEntertainment'])
        on_board_service = int(request.form['OnBoardService'])
        leg_room_service = int(request.form['LegRoomService'])
        baggage_handling = int(request.form['BaggageHandling'])
        checkin_service = int(request.form['CheckinService'])
        inflight_service = int(request.form['InflightService'])
        cleanliness = int(request.form['Cleanliness'])
        departure_delay_in_minutes = float(request.form['DepartureDelayInMinutes'])
        arrival_delay_in_minutes = float(request.form['ArrivalDelayInMinutes'])

        # Prepare data for prediction
        features = [[
            gender, customer_type, age, type_of_travel, flight_class, flight_distance,
            inflight_wifi_service, departure_arrival_time_convenient, ease_of_online_booking,
            gate_location, food_and_drink, online_boarding, seat_comfort, inflight_entertainment,
            on_board_service, leg_room_service, baggage_handling, checkin_service, inflight_service,
            cleanliness, departure_delay_in_minutes, arrival_delay_in_minutes
        ]]

        # Make the prediction
        prediction = model.predict(features)[0]

        return render_template('form.html', prediction=prediction)
    except Exception as e:
        print(f"Error during prediction:{e}")
        return render_template('form.html', prediction="Prediction error")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)