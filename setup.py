from setuptools import setup, find_packages

setup(
    name='zmongo_retriever',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'python-dotenv',
        'pymongo',
        'motor',
        'pandas',
        'openai',
        'beautifulsoup4',
        'redis',
        'nest_asyncio',
        'nbformat',
        'langchain-community',
        'chromadb',
        'langchain-ollama',
        'langchain-openai',
        'llama-cpp-python',
        'ace-tools',
    ],
    include_package_data=True,
    description='Seamless MongoDB retrieval operations using OpenAI GPT and local LLaMA models.',
    author='CentralFloridaAttorney',
    url='https://github.com/CentralFloridaAttorney/zmongo_retriever',
)
