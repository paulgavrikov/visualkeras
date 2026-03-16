import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="visualkeras",
    version="0.2.0",
    author="Paul Gavrikov",
    author_email="paul.gavrikov@hs-offenburg.de",
    description="Architecture visualization of Keras models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paulgavrikov/visualkeras",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
    ],
    install_requires=[
        'pillow>=6.2.0',
        'numpy>=1.18.1',
        'aggdraw>=1.3.11'
    ],
    python_requires='>=3.9',
)
