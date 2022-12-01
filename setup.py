from setuptools import setup, find_packages

LONG_DESCRIPTION = open("README.md", "r").read()

setup(
    name="nodeapi_utils",
    version="0.1.0",
    author="web3analytic",
    author_email="mike@paretolabs.xyz",
    packages=find_packages(),
    scripts=[],
    description="Toolkit for parallel API requests to fetch transactions.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/web3analytic-xyz/nodeapi-utils",
    install_requires=[
        "jsonlines==3.1.0",
        "numpy==1.21.4",
        "protobuf==4.21.10",
        "requests==2.28.1",
        "setuptools==41.2.0",
        "tqdm==4.62.3"
    ]
)
