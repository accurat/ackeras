from flask import Flask, make_response, request, render_template
import csv
import webbrowser
from auto_ml import Pipeline
import pandas as pd

app = Flask(__name__)


@app.route('/')
def form():
    return """ Data processing
      <form action="/transform" method="post" enctype="multipart/form-data">
        <input type="file" name="data_file" />
        <input type="submit" />
      <form>"""


if __name__ == "__main__":
    port = 5000
    app.run(port=port, debug=True)
