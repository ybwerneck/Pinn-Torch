from setuptools import setup, find_packages

setup(
    name="fisiocomPinn",  # Name that will be used to install (pip install my_library)
    version="0.1.0",  # Version
    packages=find_packages(),  # Automatically find your folders/modules
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "h5py>=3.8",
        "chaospy>=4.3",
    ],  # If you have external libraries to install (e.g., numpy)
    author="Yan Werneck, Thiago Esterci",  # Optional
    author_email="you@example.com, thiago.esterci@estudante.ufjf.br",  # Optional
    description="A cool custom PINN library",  # Optional
    long_description=open("README.md").read(),  # Optional
    long_description_content_type="text/markdown",
    url="https://github.com/ybwerneck/Pinn-Torch/",  # Optional
    classifiers=[  # Optional - for package info
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
