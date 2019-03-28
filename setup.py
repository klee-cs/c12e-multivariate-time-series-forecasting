from setuptools import setup

setup(
    name='multi-tsf',
    packages=['multi_tsf'],
    python_requires='>= 3.6, < 4.0',
    tests_require=[
        'pytest >= 3.8.1, < 4.0'
    ],
    install_requires=['absl-py>=0.7.0',
                      'astor>=0.7.1',
                      'cycler>=0.10.0',
                      'gast>=0.2.2',
                      'grpcio>=1.19.0',
                      'h5py>=2.9.0',
                      'keras>=2.2.4',
                      'keras-applications>=1.0.7',
                      'keras-preprocessing>=1.0.9',
                      'kiwisolver>=1.0.1',
                      'markdown>=3.0.1',
                      'matplotlib>=3.0.2',
                      'mock>=2.0.0',
                      'numpy>=1.16.2',
                      'pandas>=0.24.1',
                      'pbr>=5.1.2',
                      'protobuf>=3.6.1',
                      'pyparsing>=2.3.1',
                      'python-dateutil>=2.8.0',
                      'pytz>=2018.9',
                      'pyyaml>=5.1',
                      'scikit-learn>=0.20.2',
                      'scipy>=1.2.1',
                      'seaborn>=0.9.0',
                      'sklearn>=0.0',
                      'tensorboard>=1.13.0',
                      'tensorflow>=1.13.1',
                      'tensorflow-estimator>=1.13.0',
                      'tensorflow-probability>=0.6.0',
                      'termcolor>=1.1.0',
                      'tqdm>=4.31.1',
                      'werkzeug>=0.14.1'
    ]
)