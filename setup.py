import setuptools

packages = ['metagrad']

setuptools.setup(
    name='metagrad',
	version='0.1.0',
	author='Junhao Zhou',
    author_email="jhzhou.ai@gmail.com",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',)

