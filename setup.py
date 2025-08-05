from setuptools import setup

setup(
    name="HFtopodiff", 
    py_modules=["HFtopodiff"], 
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "sklearn", "opencv-python"], 
)
