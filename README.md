![ACKERAS](/frontend/ackeras.png)
===========

## Installation

The library is not pip-installable obviously but it should run with
```
    $ python ./setup.py
```

**Note:** As autokeras the library is only compatible with: **Python 3.6**.

**Note double down:** I refer to the Keras setup at [Keras](https://keras.io/) and suggest the [Theano](https://github.com/Theano/Theano) backend but you can also use [Tensorflow](https://www.tensorflow.org/api_guides/python/)

## Usage

The usage should be tailored with the pipeline.py file as follows:
``` python 
from pipeline import Pipeline
test_params = {'path': './your_file.csv',
               'categorical_feautures': ['Ship Mode', 'Country', 'Segment', 'Category', 'Sub-Category'],
               'timecolumn': 'Ship Date',
               'drop_rest': True,
               'extreme_drop': 'Row ID',
               'supervised': True,
               'reg_class': 'classification'}

plp = Pipeline(**test_params)

```

If you want you can use the classes individually as follows:
``` python 
from data_cleaning import AccuratPreprocess

acp = AccuratPreprocess(path=path)
data_processed = acp.fit_transform()

```

Note that the pipeline requiers to specify the categorical and datetime columns, and a pandas.DataFrame format so do a lot of:

``` python
import pandas as pd
assert isinstance(my_df, pd.DataFrame)
#...
```

if you decide to use it.
