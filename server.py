import csv
import webbrowser
import pandas as pd
import json
import pdb
import asyncio
import time
from multiprocessing.pool import ThreadPool
import io
from flask_jsonpify import jsonpify
import zipfile

from datetime import datetime
import uuid
from flask import Flask, make_response, request, render_template, make_response, send_from_directory, jsonify, send_file
from auto_ml import Pipeline
from bson.json_util import dumps


class Job():
    def __init__(self, data, params, pool):
        self.results = None
        self.id = str(uuid.uuid4())
        self.pipeline = Pipeline(data, **params)
        self.pool = pool

    def process(self):
        process = self.pipeline.process
        self.results = self.pool.apply_async(process, ())
        return self.results

    def status(self):
        status = self.pipeline.status
        while status != 'Done':
            payload = {'outputs': status}
            return payload

        returning = self.pipeline.outputs
        return returning

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
            string_data, params = json.loads(get_data).values()
            data = pd.read_csv(io.StringIO(string_data))
            data.columns = pd.Series(data.columns).apply(
                lambda x: x.lower().replace(' ', '-'))
            pool = ThreadPool(processes=1)
            job_instance = Job(data, params, pool)
            self.jobs.append(job_instance)
            job_instance.process()
            return job_instance.to_response()

        @self.app.route('/result/<job_id>')
        def elaborate_data(job_id):
            try:
                job = [j for j in self.jobs if j.id == job_id][0]
            except IndexError:
                job = None
                print('Not found')
                return 'This job is closed'

            output = job.status()
            output = jsonpify(output)
            if len(output == 1):
                return dumps(output)

            else:
                acp = send_file(output['aco'])
                cluster_data = send_file(output['cluster_data'])
                coefficients = send_file(output['coefficients'])

                file_list = [acp, cluster_data, coefficients]

                with zipfile.ZipFile(f'zipped_data_{id}', 'w') as zf:
                    zf.write(file_list)

                return send_file('zipped_data.zip', attachment_filename=f'zipped_data_{id}', as_attachment=True)

        self.app.run(port=5000, debug=True, host="0.0.0.0")


server = Server()
server.run()
