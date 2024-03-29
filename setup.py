import setuptools

setuptools.setup(
    name="sratta",
    version="1.0.0",
    author="Owkin",
    include_package_data=True,
    packages=setuptools.find_packages(),
    python_requires="~=3.8",
    install_requires=[
        "Pillow==9.0.1",
        "cycler==0.11.0",
        "fonttools==4.29.1",
        "kiwisolver==1.3.2",
        "lifelines==0.27.4",
        "matplotlib==3.5.1",
        "mlflow==1.23.1",
        "networkx==2.6.3",
        "numpy==1.21.5",
        "omegaconf==2.2.3",
        "openml==0.12.2",
        "packaging==21.3",
        "protobuf==3.19.0",
        "pyDeprecate==0.3.2",
        "pyparsing==3.0.7",
        "python-dateutil==2.8.2",
        "scikit-learn==1.0.2",
        "scipy==1.7.3",
        "six==1.16.0",
        "torch==1.11.0",
        "torchmetrics==0.7.2",
        "torchvision==0.12.0",
        "tqdm==4.63.0",
        "typing-extensions==4.1.1",
        "loguru==0.6.0",
    ],
)
