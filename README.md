![ACKERAS](/frontend/ackeras.png)
===========

## Installation

The library is not pip-installable obviously but it should run with
```
    $ python ./setup.py
```

**Note:** As autokeras the library is only compatible with: **Python 3.6**.

**Note double down:** I refer to the Keras setup at [Keras](https://keras.io/) and suggest the [Theano](https://github.com/Theano/Theano) backend but you can also use [Tensorflow](https://www.tensorflow.org/api_guides/python/)

## Disclaimer

It just started so most things do not work properly or need to be fix, there are plenty of #TODO inside, but feel free to use and to pull.

## Scope

The idea is to be able to input a file in CSV or JSON format and, after selecting a few parameters (see below), getting your data cleaned and clustered automatically, ready to be analyzed. This can be useful in the context of preliminary analysis and to implement some outputs in visualization (e.g. a clustering in a scatterplot or the probabilities of a certain class with a decision tree etc.).

The implementations are:
- [x] Data cleaning: NaN filling with various methods, label encoding and one hot encoding, flagging of categorical feautures and dropping redundant feautures (almost);
- [x] Dimensionality Reduction: [PCA](http://setosa.io/ev/principal-component-analysis/) and [UMAP](https://github.com/lmcinnes/umap)
- [x] Clustering: [k-means](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/), with silhoutte analysis optimization, and [DBSCAN](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/) clustering;
- [x] Logistic and Linear regression, with K-fold cross validation.
- [ ] [Random Forests](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) and [Support Vector Machines](https://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html), with genetic algorithm optimization.
- [ ] [Neural Networks](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.88343&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false), with Auto-Keras
- [ ] ML visualizations with Seaborn and Lime

## Parameters

The parameters of the class are:

- input_data: a pd.DataFrame with the data input
- categorical_feautures: a list of categorical feautures
- timecolumn: the datetime columns name
- extreme_drop: drop this column in a worst case scenario fashion, usually it can be None
- y: the dependent variable in supervised problems
- drop_rest: keep it True
- supervised: whether the problem is supervised or unsupervised

## Usage

You shold now be able to interact with the dataset through a simple server that is only running on my machine in the local network now. Fixing is happening anyhow so stay tuned. To test it yourself just try:

```
cd ackeras
$ python server.py
```

and head over to your localhost:5000. Upload a CSV and you should see something like this:

![test](/frontend/mock.png)

Be sure to tick (at this stage) the "Drop_rest", because it ensures that the data you push in and is not understood will be excluded. Then go ahead and submit query and head over to the link provided and enjoy everything breaking down. Keep an eye on the console because we tried and log most errors.

## Usage with python

The usage should be tailored with the pipeline.py file as follows:
``` python 
from pipeline import Pipeline
test_params = {'path': './your_file.csv',
               'categorical_feautures': ['Ship Mode', 'Country', 'Segment', 'Category', 'Sub-Category'],
               'timecolumn': 'Ship Date',
               'drop_rest': True,
               'extreme_drop': 'Row ID',
               'supervised': True,
               'reg_class': (None, 'Country')}

plp = Pipeline(**test_params)

```

If you want you can use the classes individually as follows:
``` python 
from data_cleaning import AccuratPreprocess

acp = AccuratPreprocess(path=path)
data_processed = acp.fit_transform()

```

#### Other interesting libraries to add in the pipeline

- [Awesome Dash](https://github.com/Acrotrend/awesome-dash), python + react.js + flask
- [Bokeh](https://github.com/bokeh/bokeh), interactive web-plotting

