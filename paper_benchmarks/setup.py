from setuptools import find_packages, setup
import os
import glob  

package_name = 'paper_benchmarks'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # installa TUTTI i file .launch.py nella cartella launch
        (os.path.join('share', package_name, 'launch'),
            glob.glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'),
            ['urdf/assembly_with_hand.urdf', 'urdf/assembly.urdf']),
        (os.path.join('share', package_name, 'config'),
            ['config/dynamics_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mbari',
    maintainer_email='bari.marcello00@gmail.com',
    description='Benchmark tools for digital twin paper',
    license='GPL-3.0-only',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'benchmark_dynamics = paper_benchmarks.benchmark_dynamics:main',
        ],
    },
)