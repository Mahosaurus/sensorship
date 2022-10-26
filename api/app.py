from datetime import datetime
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
app = Flask(__name__)

@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/sensor-data', methods=['POST'])
def hello():
    if request.method == 'POST':
        data = request.json
        return f"Received {data}"

if __name__ == '__main__':
   app.run()
