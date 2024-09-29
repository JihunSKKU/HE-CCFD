from setuptools import setup, find_packages

setup(
    name='HE-CCFD',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'pandas',
        'scikit-learn',
        'numpy',
        # another dependencies
    ],
)
