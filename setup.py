from setuptools import setup, find_packages


# with open('README.rst') as f:
#     readme = f.read()

# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='imgproc',
    version='0.1.alpha',
    description='Image processing library',
    long_description='',
    author='Johan Ferret',
    author_email='johan.ferret1@gmail.com',
    url='https://github.com/ferretj/',
    license='',
    packages=find_packages(exclude=('tests', 'docs'))
)