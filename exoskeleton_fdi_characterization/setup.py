from setuptools import setup
from glob import glob
import os

package_name = 'exoskeleton_fdi_characterization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='FDI maintainer',
    maintainer_email='dev@example.com',
    description='FDI Phase 0 — characterization infrastructure.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'regime_scheduler   = exoskeleton_fdi_characterization.regime_scheduler:main',
            'fault_orchestrator = exoskeleton_fdi_characterization.fault_orchestrator:main',
        ],
    },
)