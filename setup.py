from setuptools import setup, find_packages


# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()


setup(
    name='improcrf',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here
        ],
    },
    author='Elliot London',
    author_email='elliot.london@rfpro.com',
    description='Python image processing library for manipulating, assessing and comparing simulated and real images',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/elliotl-rfpro/ImageProcessing',
    classifiers=[
        'Python',
    ],
    python_requires='>=3.9',
)
