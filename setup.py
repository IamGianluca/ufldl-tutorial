from setuptools import find_packages, setup


setup(
    name='ufldl',
    version='0.2.0',
    description='Python Machine Learning library',
    author='Gianluca Rossi',
    author_email='gr.gianlucarossid@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        'ufldl': ['datasets/data/*.csv', 'datasets/data/*.gz',
                  'datasets/data/*.p']
    },
    license='MIT License'
)
