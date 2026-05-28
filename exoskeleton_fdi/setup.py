from setuptools import find_packages, setup

package_name = 'exoskeleton_fdi'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/fdi_params.yaml']),
        ('share/' + package_name + '/launch', ['launch/fdi.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mbari',
    maintainer_email='mbari@todo.todo',
    description='Standalone Fault Detection and Isolation for exoskeleton sensors',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fdi_node = exoskeleton_fdi.fdi_node:main',
        ],
    },
)
