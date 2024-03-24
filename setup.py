from setuptools import setup,find_packages
from typing import List

HYPER = "-e ."
def get_requirements(file_path: str)->List[str]:

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements ]
    if HYPER in requirements:
        requirements.remove(HYPER)

setup(
    name='serious_one',
    version='0.0.0',
    author='Prince',
    author_email='prince991151@gmail.com',
    description="serious_one",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')

    
)