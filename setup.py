from distutils.core import setup

setup(
    name='ackeras',
    packages=['ackeras'],  # this must be the same as the name above
    install_requires=['torch==0.4.0', 'torchvision==0.2.1', 'numpy==1.14.5',
                      'keras==2.2.2', 'scikit-learn==0.19.1', 'tensorflow', 'autokeras'],
    author='Andrea Titton',
    author_email='andreatitton96@gmail.com',
    keywords=['automl'],  # arbitrary keywords
    classifiers=[]
)
