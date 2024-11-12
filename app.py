from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask Server"

@app.route('/api/data', methods=['POST'])
def post_data():
    # Get all form data as a dictionary
    form_data = request.form

    # List of all expected form variables:
    # - year
    # - month
    # - carrier
    # - carrier_name
    # - airport
    # - airport_name
    # - arr_flights
    # - arr_del15
    # - carrier_ct
    # - weather_ct
    # - nas_ct
    # - security_ct
    # - late_aircraft_ct
    # - arr_cancelled
    # - arr_diverted
    # - arr_delay
    # - carrier_delay
    # - weather_delay
    # - nas_delay
    # - security_delay
    # - late_aircraft_delay

    # Example: To extract any field from the form data:
    # carrier = form_data.get('carrier')  # This extracts the 'carrier' field from the form

    # Return the entire form data as response (for debugging or processing)
    response = {
        "message": "Data received successfully!",
        "data": form_data  # Return all the form data as received
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
