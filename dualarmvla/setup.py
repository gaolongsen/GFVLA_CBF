import pathlib
import pkg_resources
from setuptools import find_packages, setup


def _read_install_requires():
    with pathlib.Path('requirements.txt').open() as fp:
        packages = [str(requirement) for requirement in pkg_resources.parse_requirements(fp)]
        return packages


setup(
    name='dualarmvla',
    version='1.0',
    author='CSCSX',
    author_email='longsengao@gmail.com',
    description='LVA information fusion for robotic manipulation - Longsen Gao & Shunlei Li',
    long_description=pathlib.Path('README.md').open().read(),
    long_description_content_type='text/markdown',
    keywords=[
        'Robotics',
        'Embodied Intelligence',
        'Representation Learning',
    ],
    license='MIT License',
    packages=find_packages(include='dualarmvla.*'),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    python_requires='>=3.9',
)