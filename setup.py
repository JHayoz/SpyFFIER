from distutils.core import setup

setup(
    name='spyffier',
    version='0.1.0',
    packages=['spyffier'],
    url='https://github.com/JHayoz/SpyFFIER',
    license='MIT',
    author='Jean Hayoz',
    author_email='jeanhayoz94@gmail.com',
    description='Python package to reduce VLT/ERIS/SPIFFIER data using esorex commands and custom functionalities.',
    install_requires=[
                'numpy',
                'scipy',
                'matplotlib',
                'astropy',
                'PyAstronomy',
                'pandas',
                'sklearn',
                'seaborn',
                'spectres',
                'photutils',
                'skycalc_ipy'],
)
