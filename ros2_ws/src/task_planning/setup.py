"""
setup.py - Python Package Configuration for Task Planning
ML-Based Object Picking Robot - Task Planning Module

This setup.py configures the task_planning ROS2 Python package for:
- High-level task coordination
- Pick-and-place sequence planning
- State machine implementation
- Safety monitoring and validation
- Task execution and monitoring

Author: Victor's Production Team
Date: 2025-01-12
Version: 1.0.0

Usage:
    colcon build --packages-select task_planning
"""

from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'task_planning'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Install package marker
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        
        # Install package.xml
        ('share/' + package_name, ['package.xml']),
        
        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        
        # Install configuration files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        
        # Install state machine definitions
        (os.path.join('share', package_name, 'config', 'state_machines'),
            glob('config/state_machines/*.yaml')),
        
        # Install task sequences
        (os.path.join('share', package_name, 'config', 'tasks'),
            glob('config/tasks/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'pyyaml>=5.4.0',
    ],
    zip_safe=True,
    
    # Metadata
    author='Victor\'s Production Team',
    author_email='victor@production-team.com',
    maintainer='Victor\'s Production Team',
    maintainer_email='victor@production-team.com',
    description='High-level task planning and coordination for ML-based object picking robot',
    license='Proprietary',
    
    # Testing
    tests_require=['pytest'],
    
    # Entry points - ROS2 node executables
    entry_points={
        'console_scripts': [
            # Main task planning node
            'task_planner_node = task_planning.task_planner_node:main',
            
            # Pick and place coordinator
            'pick_place_planner = task_planning.pick_place_planner:main',
            
            # State machine executor
            'state_machine_node = task_planning.state_machine:main',
            
            # Safety monitor
            'safety_monitor = task_planning.safety_monitor:main',
            
            # Task executor
            'task_executor = task_planning.task_executor:main',
            
            # Sequence coordinator
            'sequence_coordinator = task_planning.sequence_coordinator:main',
        ],
    },
    
    # Package classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Robotics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Framework :: Robot Operating System 2',
    ],
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Additional keywords for searchability
    keywords=[
        'ros2',
        'robotics',
        'task-planning',
        'pick-and-place',
        'state-machine',
        'manipulation',
        'automation',
        'machine-learning',
    ],
    
    # Project URLs
    project_urls={
        'Source': 'https://github.com/victor/ml-object-picking-robot',
        'Bug Reports': 'https://github.com/victor/ml-object-picking-robot/issues',
        'Documentation': 'https://ml-robot-docs.example.com',
    },
)
