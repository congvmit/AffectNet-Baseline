from setuptools import setup, find_packages

setup(
    name='affectnet',
    packages=find_packages(exclude=['examples']),
    version='0.1.0',
    license='MIT',
    description='AffectNet - Pytorch',
    author='Cong M. Vo',
    author_email='congvm.it@gmail.com',
    keywords=[
        'artificial intelligence',
        'attention mechanism',
        'image recognition'
    ],
    install_requires=[
        'torch>=1.6',
        'einops>=0.3'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
