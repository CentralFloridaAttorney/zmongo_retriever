from setuptools import setup, find_packages

setup(
    name='zmongo_retriever',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies from requirements.txt or inline here
    ],
    include_package_data=True,
    description='Seamless MongoDB retrieval operations using OpenAI GPT.',
    author='CentralFloridaAttorney',
    url='https://github.com/CentralFloridaAttorney/zmongo_retriever',
)
