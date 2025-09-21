from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='ml_package',
    version='0.1.0',
    author='Deadshot',
    author_email='rempire046@gmail.com',
    description='A machine learning package for various algorithms',
    packages=find_packages(),
    install_requires= get_requirements('requirement.txt'),
)