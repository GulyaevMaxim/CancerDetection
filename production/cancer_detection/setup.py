"""liveface setup script."""
from setuptools import setup, find_packages
from subprocess import check_output
import re


# Check Git LFS available
try:
    out = check_output(['git', 'lfs', 'env']).decode("utf-8")

    lfs_test = re.compile(
        r'^git config filter\.lfs\.(\S+)\s+=\s+\"([^\"]+)\"',
        re.MULTILINE | re.IGNORECASE
    )
    print(out)

    lfs_setup = {k: len(v.strip()) > 0 for k, v in lfs_test.findall(out)}
    # check LFS has correct setup (it has no setup if not installed)
    if (lfs_setup.get('clean', False) is not True or
            lfs_setup.get('smudge', False) is not True or
            lfs_setup.get('process', False) is not True):
        raise EnvironmentError(
            'Git LFS is not configured. Run "git lfs install".')
except Exception:
    raise EnvironmentError(
        'Git LFS is required to install this package. Please install Git LFS '
        '(https://git-lfs.github.com/) and restart package installation.'
    )


# Bad hack
# parse_requirements() returns generator of pip.req.InstallRequirement objects
# install_reqs = parse_requirements("requirements.txt", session='hack')
# requirements_bad = [str(ir.req) for ir in install_reqs]

with open('requirements.txt') as fp:
    install_requires = fp.read()

setup(
    name='cancerdetection',
    version='0.0.1',
    install_requires=install_requires,
    description='Cancer detection on images',
    author='MaximGulyaev',
    author_email='gulyaev@cvisionlab.com',
    packages=find_packages(),
    package_data={'cancerdetection': []},
    entry_points={
        'console_scripts': [
		
        ]
    },
    zip_safe=False
)
