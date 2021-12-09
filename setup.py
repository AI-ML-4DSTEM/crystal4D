import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crystal4D",                     # This is the name of the package
    version="0.1.4",                        # The initial release version
    author="Joydeep Munshi",                     # Full name of the author
    description="Deep learning useful information from diffraction images",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    keywords="STEM 4DSTEM AI/ML",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["crystal4D"],             # Name of the python package
    package_dir={'':'./'},     # Directory of the source code of the package
    install_requires=[
        'numpy >= 1.19',
        'matplotlib >= 3.4.2']
)
