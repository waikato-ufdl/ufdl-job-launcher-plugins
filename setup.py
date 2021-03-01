from setuptools import setup, find_namespace_packages

def _read(f) -> bytes:
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="ufdl.joblauncher-plugins",
    description="Plugins for the ufdl-job-launcher framework.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/waikato-ufdl/ufdl-job-launcher-plugins",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3',
    ],
    license='Apache 2.0 License',
    package_dir={
        '': 'src'
    },
    packages=find_namespace_packages(where='src'),
    namespace_packages=[
        "ufdl",
    ],
    version="0.0.1",
    author='Peter Reutemann',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "wai.annotations>=0.4.1",
        "ufdl.pythonclient",
        "ufdl.json-messages",
        "ufdl.joblauncher",
        "tensorflow",
    ],
)
