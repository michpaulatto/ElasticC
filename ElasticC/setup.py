import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="elasticc-michpaulatto", # Replace with your own username
    version="0.0.1",
    author="Michele Paulatto",
    author_email="mpaulatt@imperial.ac.uk",
    description="Effective elastic properties calculator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/michpaulatto/ElasticC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL 4.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
