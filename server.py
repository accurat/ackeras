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
            params['insample'] = int(params['insample'])
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
                return 'This job is closed'
            output = job.status()
            output = jsonpify(output)
            try:
                progress = json.loads(
                    output.get_data().decode('utf-8'))['outputs']
            except KeyError:
                progress = None

            if progress == 'Working...':
                return dumps(progress)

            else:
                output_data = json.loads(output.get_data().decode('utf-8'))
                buff = io.BytesIO()
                zip_archive = zipfile.ZipFile(buff, mode='w')

                for k, values in output_data.items():
                    file_buff = io.StringIO()
                    file_buff.write(dumps(values))
                    zip_archive.writestr(
                        f'{k}_{job_id}.json', file_buff.getvalue())

                zip_archive.close()
                buff.seek(0)

                return send_file(buff, attachment_filename=f'zipped_data_{job_id}.zip', as_attachment=True)

        self.app.run(port=5000, debug=True, host="0.0.0.0")


server = Server()
server.run()
