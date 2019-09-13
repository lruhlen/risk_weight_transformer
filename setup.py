from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='risk_weight_transformer',
    version='0.1.0',
    description='Scikit-learn Pipeline compatible risk-weight transformer',
    long_description=readme,
    author='Laurel Ruhlen',
    author_email='ruhlen@gmail.com',
    url='https://github.com/lruhlen/risk_weight_transformer',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
