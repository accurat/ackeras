import csv
import webbrowser
import pandas as pd
import json
import pdb
import asyncio
import time
from multiprocessing.pool import ThreadPool

from datetime import datetime
from flask import Flask, make_response, request, render_template, make_response, send_from_directory, jsonify
from auto_ml import Pipeline
from bson.json_util import dumps


class Job():
    def __init__(self, data, params, pool):
        self.results = None
        self.id = int(time.mktime(datetime.now().timetuple()))
        self.pipeline = Pipeline(data, **params)
        self.pool = pool

    def process(self):
        process = self.pipeline.process
        self.results = self.pool.apply_async(process, ())
        return self.results

    def to_response(self):
        payload = {'id': self.id}
        return dumps(payload)


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
            data, params = json.loads(get_data).values()
            pool = ThreadPool(processes=1)
            job_instance = Job(data, params, pool)
            self.jobs.append(job_instance)
            return job_instance.to_response()

        @self.app.route('/result/<job_id>')
        def elaborate_data(job_id):
            try:
                job = [j for j in self.jobs if j.id == job_id][0]
            except IndexError:
                job = None
                print('Not found')
                return job

            output = job.results.get()
            if output is not None:
                assert isinstance(output, dict)
                response = make_response(output)
                response.headers['Content-Disposition'] = 'attachment; filename=result.json'
                response.headers['Content-Type'] = "text/csv"
                return response

            else:
                return render_template('not_ready.html')

        self.app.run(port=5000, debug=True)


server = Server()
server.run()
