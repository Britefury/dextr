import os
from setuptools import find_packages
from setuptools import setup

version = '0.1.2'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.md')).read()
except IOError:
    README = ''

install_requires = [
    'numpy',
    'scipy',
    'Pillow',
    'scikit-image',
    'torch',
    'torchvision',
    ]

tests_require = [
    ]

include_package_data = True

setup(
    name="dextr",
    version=version,
    description="PyTorch Deep Extreme Cut library",
    long_description="\n\n".join([README]),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="",
    author="Geoff French",
    url="https://github.com/Britefury/dextr",
    license="MIT",
    packages=find_packages(),
    include_package_data=include_package_data,
    zip_safe=False,
    install_requires=install_requires,
    extras_require={
        'testing': tests_require,
        },
    )
