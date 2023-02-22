"""setup for gw_eccentricity."""
import setuptools

with open("README.md", "rb") as fh:
    long_description = fh.read().decode("UTF-8")


# Extract code version from gw_eccentricity.py
def get_version():
    """Get package version."""
    with open('gw_eccentricity/gw_eccentricity.py') as f:
        for line in f.readlines():
            if "__version__" in line:
                return line.split('"')[1]


setuptools.setup(
    name="gw_eccentricity",
    version=get_version(),
    author="Md Arif Shaikh, Vijay Varma, Harald Pfeiffer",
    author_email="arifshaikh.astro@gmail.com, vijay.varma392@gmail.com",
    description="Defining eccentricity for gravitational wave astronomy.",
    keywords='black-holes gravitational-waves',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vijayvarma392/gw_eccentricity",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'lalsuite',
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
