from setuptools import setup, find_packages

setup(
    name="dream11",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask==2.3.3',
        'pandas==2.1.4',
        'numpy==1.26.3',
        'scikit-learn==1.3.2',
        'catboost==1.2.2',
        'xgboost==2.0.3',
        'pulp==2.7.0',
        'python-dotenv==1.0.0',
        'gunicorn==21.2.0'
    ]
) 