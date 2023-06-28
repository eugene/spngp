from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'First package of the SPNGP project'
# Setting up
setup(
       # 'name' deve corresponder ao nome da pasta 'verysimplemodule'
        name="projectspngp", 
        version=VERSION,
        author="Maria Eduarda Barbosa",
        author_email="m4dud01@gmail.com",
        description=DESCRIPTION,
        packages=find_packages(),        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.10",
        ],
)