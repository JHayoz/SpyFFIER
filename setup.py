#!/usr/bin/env python

import pkg_resources
import setuptools

with open('requirements.txt') as req_txt:
    parse_req = pkg_resources.parse_requirements(req_txt)
    install_requires = [str(req) for req in parse_req]

setuptools.setup(
    name='spyffier',
    version='0.1.0',
    description='Data reduction pipeline for VLT/ERIS/SPIFFIER',
    long_description=open('README.md').read(),
    long_description_content_type='text/x-rst',
    author='Jean Hayoz',
    author_email='jhayoz@phys.ethz.ch',
    url='https://github.com/JHayoz/SpyFFIER',
    project_urls={},
    packages=['spyffier'],
    package_data={'spyffier': ['*.txt']},
    install_requires=install_requires,
    tests_require=[],
    license='GNU',
    zip_safe=False,
    keywords='spyffier',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
    ],
)
