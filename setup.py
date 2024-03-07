from setuptools import setup, find_packages

setup(
    name='zmongo_retriever',
    version='0.1.0',
    author='John M. Iriye',
    author_email='iriye@yahoo.com',
    description='A utility for retrieving documents from MongoDB collections.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/centralfloridaattorney/zmongo_retriever',
    packages=find_packages(),
    install_requires=[
        'pymongo',
        'langchain',
        'langchain_community',
        'langchain_text_splitters',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
