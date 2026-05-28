from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f
                    if line.strip() and not line.startswith("#") and not line.startswith("-r")]

setup(
    name="metric-reliability-summarization",
    version="1.0.0",
    author="Varun D Kanchan, Abhishek Shetty",
    author_email="kanchanvarun45@gmail.com",
    description="Joint analysis of human alignment and stochastic stability for summarization metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/metric-reliability-summarization",
    packages=find_packages(where="scripts"),
    python_requires="==3.10.*",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="summarization metrics nlp evaluation bert-score bleu rouge meteor comet bleurt",
)
