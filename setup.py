import setuptools

with open("README.md", "rb") as fh:
    long_description = fh.read().decode("UTF-8")

# Extract code version from __init__.py
def get_version():
    with open('measureEccentricity/measureEccentricity.py') as f:
        for line in f.readlines():
            if "__version__" in line:
                return line.split('"')[1]

setuptools.setup(
    name="measureEccentricity",
    version=get_version(),
    author="Md Arif Shaikh, Vijay Varma",
    author_email="arif.shaikh@icts.res.in, vijay.varma392@gmail.com",
    description="Measure eccentricity from gravitational waves.",
    keywords='black-holes gravitational-waves',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vijayvarma392/Eccentricity",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'lalsuite',
        'gwtools',
        'palettable'
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
    ],
)
