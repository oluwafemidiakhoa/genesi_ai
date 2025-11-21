"""
Genesis RNA - RNA Foundation Model for Cancer Research
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='genesis-rna',
    version='0.1.0',
    author='GENESI AI Team',
    author_email='noreply@example.com',
    description='Genesis RNA - RNA foundation model for breast cancer cure research',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/oluwafemidiakhoa/genesi_ai',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'cancer': [
            'scikit-learn>=1.0.0',
            'scipy>=1.7.0',
            'plotly>=5.0.0',
        ],
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'genesis-train=genesis_rna.train_pretrain:main',
        ],
    },
    include_package_data=True,
    package_data={
        'genesis_rna': ['*.yaml', '*.json'],
    },
    project_urls={
        'Bug Reports': 'https://github.com/oluwafemidiakhoa/genesi_ai/issues',
        'Source': 'https://github.com/oluwafemidiakhoa/genesi_ai',
        'Documentation': 'https://github.com/oluwafemidiakhoa/genesi_ai/blob/main/README.md',
    },
    keywords='rna deep-learning transformer cancer genomics bioinformatics',
    zip_safe=False,
)
