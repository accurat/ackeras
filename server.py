import csv
import webbrowser
import pandas as pd
import json
import pdb
from flask import Flask, make_response, request, render_template, send_from_directory
from auto_ml import Pipeline
from bson.json_util import dumps


class Server():
    def __init__(self):
        self.app = Flask(__name__, template_folder='frontend')
        self.jobs = []

    def run(self):

        @self.app.route('/', methods=['GET'])
        def home():
            return render_template('index.html')

        @self.app.route('/static/<path>', methods=['GET'])
        def server_static(path):
            return send_from_directory('frontend', path)

        @self.app.route('/config', methods=['POST'])
        def get_data():
            get_data = request.data.decode()
            data, params = json.loads(get_data)
            self.jobs.append(data, params)

            return "Data loaded"

        self.app.run(port=5000, debug=True)


server = Server()
server.run()
