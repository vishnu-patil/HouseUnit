from flask import Flask, render_template, request, jsonify
from utils import House
import numpy as np

app = Flask(__name__)

@app.route('/')
def main():
    print("This Is Home Page")
    return render_template("home.html")

@app.route('/house_price', methods = ['POST','GET'])
def house_price():
    if request.method == 'POST':
        data = request.form
        print(data)

        house = House(data)
        price = house.predict()
        # return jsonify({"Price Of House":price})
        return render_template("home.html", prediction = price)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5555)        


