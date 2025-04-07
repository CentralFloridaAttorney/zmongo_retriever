from setuptools import setup, find_packages

setup(
    name='zmongo_retriever',
    version='0.1.4',
    packages=find_packages(),
    install_requires=[
        'python-dotenv',
        'pymongo',
        'motor',
        'pandas',
        'openai',
        'beautifulsoup4',
        'nest_asyncio',
        'nbformat',
        'chromadb',
        'langchain-ollama',
        'langchain-openai',
        'llama-cpp-python',
        'ace-tools',
        'matplotlib',
        'opencv-python',
        'mtcnn'
    ],
    include_package_data=True,
    description='Seamless MongoDB retrieval operations using OpenAI GPT and local LLaMA models.',
    author='CentralFloridaAttorney',
    url='https://github.com/CentralFloridaAttorney/zmongo_retriever',
)
