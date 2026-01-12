#!/usr/bin/env python3
"""
Setup script for system_monitor ROS2 package.
Production-grade deployment configuration with comprehensive dependency management.
"""

from setuptools import setup
import os
from glob import glob
from pathlib import Path
import sys

# Check Python version
if sys.version_info < (3, 8):
    print("Python 3.8 or higher is required")
    sys.exit(1)

# Read package version from version file
def get_version():
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "4.1.0"

# Read long description from README
def get_long_description():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text()
    return "Production-grade system monitoring for robotic systems"

# Read requirements from requirements.txt
def get_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        return requirements_file.read_text().splitlines()
    return []

# Get all data files
def get_data_files():
    data_files = []
    
    # Package resources
    data_files.append(('share/ament_index/resource_index/packages', ['resource/system_monitor']))
    
    # Package.xml
    data_files.append(('share/system_monitor', ['package.xml']))
    
    # Launch files
    launch_files = []
    for launch_pattern in ['launch/*.launch.py', 'launch/**/*.launch.py']:
        launch_files.extend(glob(launch_pattern, recursive=True))
    if launch_files:
        data_files.append(('share/system_monitor/launch', launch_files))
    
    # Config files
    config_files = []
    for config_pattern in ['config/*.yaml', 'config/*.yml', 'config/**/*.yaml', 'config/**/*.yml']:
        config_files.extend(glob(config_pattern, recursive=True))
    if config_files:
        data_files.append(('share/system_monitor/config', config_files))
    
    # Scripts
    script_files = []
    for script_pattern in ['scripts/*.sh', 'scripts/*.py']:
        script_files.extend(glob(script_pattern))
    if script_files:
        data_files.append(('share/system_monitor/scripts', script_files))
    
    # Systemd service files
    systemd_files = []
    for systemd_pattern in ['systemd/*.service', 'systemd/*.timer']:
        systemd_files.extend(glob(systemd_pattern))
    if systemd_files:
        data_files.append(('share/system_monitor/systemd', systemd_files))
    
    # Logrotate configuration
    logrotate_files = []
    for logrotate_pattern in ['logrotate/*']:
        logrotate_files.extend(glob(logrotate_pattern))
    if logrotate_files:
        data_files.append(('share/system_monitor/logrotate', logrotate_files))
    
    # Kubernetes manifests
    k8s_files = []
    for k8s_pattern in ['kubernetes/*.yaml', 'kubernetes/**/*.yaml']:
        k8s_files.extend(glob(k8s_pattern, recursive=True))
    if k8s_files:
        data_files.append(('share/system_monitor/kubernetes', k8s_files))
    
    # Dashboard templates
    dashboard_files = []
    for dashboard_pattern in ['dashboard/*.json', 'dashboard/**/*.json']:
        dashboard_files.extend(glob(dashboard_pattern, recursive=True))
    if dashboard_files:
        data_files.append(('share/system_monitor/dashboard', dashboard_files))
    
    # Alert manager configuration
    alert_files = []
    for alert_pattern in ['alertmanager/*.yaml', 'alertmanager/**/*.yaml']:
        alert_files.extend(glob(alert_pattern, recursive=True))
    if alert_files:
        data_files.append(('share/system_monitor/alertmanager', alert_files))
    
    # Documentation
    doc_files = []
    for doc_pattern in ['docs/*.md', 'docs/**/*.md', 'docs/*.rst', 'docs/**/*.rst']:
        doc_files.extend(glob(doc_pattern, recursive=True))
    if doc_files:
        data_files.append(('share/system_monitor/docs', doc_files))
    
    # Test data
    test_data_files = []
    for test_pattern in ['test/data/*', 'test/**/data/*']:
        test_data_files.extend(glob(test_pattern, recursive=True))
    if test_data_files:
        data_files.append(('share/system_monitor/test/data', test_data_files))
    
    # Requirements file
    if Path('requirements.txt').exists():
        data_files.append(('share/system_monitor', ['requirements.txt']))
    
    # Docker files
    docker_files = []
    for docker_pattern in ['docker/*', 'docker/**/*']:
        docker_files.extend(glob(docker_pattern, recursive=True))
    if docker_files:
        data_files.append(('share/system_monitor/docker', docker_files))
    
    # CI/CD configuration
    ci_files = []
    for ci_pattern in ['.github/workflows/*.yml', '.github/**/*.yml', '.gitlab-ci.yml', 'Jenkinsfile']:
        ci_files.extend(glob(ci_pattern, recursive=True))
    if ci_files:
        data_files.append(('share/system_monitor/ci', ci_files))
    
    return data_files

# Entry points for console scripts
entry_points = {
    'console_scripts': [
        # Core monitoring nodes
        'health_checker = system_monitor.health_checker:main',
        'performance_monitor = system_monitor.performance_monitor:main',
        'logger_node = system_monitor.logger_node:main',
        
        # Utility scripts
        'system_monitor_main = system_monitor.system_monitor_main:main',
        'monitoring_dashboard = system_monitor.monitoring_dashboard:main',
        'performance_report = system_monitor.performance_report:main',
        'log_analyzer = system_monitor.log_analyzer:main',
        'alert_manager = system_monitor.alert_manager:main',
        
        # CLI tools
        'monitor-status = system_monitor.cli.status:main',
        'monitor-metrics = system_monitor.cli.metrics:main',
        'monitor-logs = system_monitor.cli.logs:main',
        'monitor-alerts = system_monitor.cli.alerts:main',
        'monitor-config = system_monitor.cli.config:main',
        
        # Maintenance tools
        'monitor-backup = system_monitor.cli.backup:main',
        'monitor-restore = system_monitor.cli.restore:main',
        'monitor-cleanup = system_monitor.cli.cleanup:main',
        'monitor-update = system_monitor.cli.update:main',
        
        # Diagnostic tools
        'monitor-diagnose = system_monitor.cli.diagnose:main',
        'monitor-benchmark = system_monitor.cli.benchmark:main',
        'monitor-profile = system_monitor.cli.profile:main',
        
        # Development tools
        'monitor-test = system_monitor.cli.test:main',
        'monitor-coverage = system_monitor.cli.coverage:main',
        'monitor-lint = system_monitor.cli.lint:main',
        
        # Integration tools
        'monitor-export = system_monitor.cli.export:main',
        'monitor-import = system_monitor.cli.import:main',
        'monitor-sync = system_monitor.cli.sync:main',
    ],
}

# Package dependencies with version ranges
install_requires = [
    # Core dependencies
    'setuptools>=58.2.0',
    
    # ROS2 Python bindings (installed separately via ROS2)
    # 'rclpy>=3.0.0',
    # 'std_msgs>=4.2.0',
    # 'sensor_msgs>=4.2.0',
    # 'geometry_msgs>=4.2.0',
    # 'diagnostic_msgs>=4.2.0',
    # 'visualization_msgs>=4.2.0',
    # 'nav_msgs>=4.2.0',
    # 'std_srvs>=4.2.0',
    
    # Data processing and numerical computing
    'numpy>=1.21.0,<2.0.0',
    'pandas>=1.5.0,<3.0.0',
    'scipy>=1.9.0,<2.0.0',
    
    # Machine learning and statistics
    'scikit-learn>=1.2.0,<2.0.0',
    'statsmodels>=0.13.0,<1.0.0',
    'joblib>=1.2.0,<2.0.0',
    
    # System monitoring
    'psutil>=5.9.0,<6.0.0',
    'gpustat>=1.0.0,<2.0.0',
    'python-apt>=2.0.0,<3.0.0',
    
    # Logging and serialization
    'msgpack>=1.0.0,<2.0.0',
    'python-json-logger>=2.0.0,<3.0.0',
    'pyyaml>=6.0,<7.0',
    'colorlog>=6.0.0,<7.0.0',
    
    # Compression and encryption
    'lz4>=4.0.0,<5.0.0',
    'brotli>=1.0.0,<2.0.0',
    'zstandard>=0.18.0,<1.0.0',
    'cryptography>=38.0.0,<42.0.0',
    
    # Visualization
    'matplotlib>=3.5.0,<4.0.0',
    'seaborn>=0.12.0,<1.0.0',
    'plotly>=5.10.0,<6.0.0',
    'bokeh>=3.0.0,<4.0.0',
    
    # Web frameworks and APIs
    'flask>=2.0.0,<3.0.0',
    'fastapi>=0.95.0,<1.0.0',
    'uvicorn>=0.21.0,<1.0.0',
    'requests>=2.28.0,<3.0.0',
    
    # Database and storage
    'redis>=4.0.0,<5.0.0',
    'influxdb-client>=1.30.0,<2.0.0',
    'prometheus-client>=0.14.0,<1.0.0',
    'elasticsearch>=8.0.0,<9.0.0',
    'pymongo>=4.0.0,<5.0.0',
    'sqlalchemy>=2.0.0,<3.0.0',
    'alembic>=1.10.0,<2.0.0',
    
    # Monitoring and observability
    'watchdog>=2.1.0,<3.0.0',
    'cachetools>=5.0.0,<6.0.0',
    'humanize>=4.0.0,<5.0.0',
    'tabulate>=0.9.0,<1.0.0',
    'tqdm>=4.64.0,<5.0.0',
    
    # Data validation and schemas
    'pydantic>=1.10.0,<2.0.0',
    'jsonschema>=4.0.0,<5.0.0',
    
    # Time and date handling
    'pytz>=2022.0',
    'python-dateutil>=2.8.0',
    
    # Networking and communication
    'websockets>=11.0.0,<12.0.0',
    'aiohttp>=3.8.0,<4.0.0',
    
    # Optional but recommended for enhanced features
    'tensorflow>=2.10.0; platform_system != "Windows"',
    'torch>=1.13.0; platform_system != "Windows"',
    'xgboost>=1.7.0; platform_system != "Windows"',
    'lightgbm>=3.3.0; platform_system != "Windows"',
    'catboost>=1.0.0; platform_system != "Windows"',
    
    # Advanced analytics
    'prophet>=1.1.0; platform_system != "Windows"',
    'pyod>=1.0.0',
    'networkx>=3.0,<4.0',
    
    # Dashboard and UI
    'dash>=2.9.0,<3.0.0',
    'dash-bootstrap-components>=1.4.0,<2.0.0',
    
    # Image processing (for visualization)
    'pillow>=9.0.0,<10.0.0',
    'opencv-python>=4.7.0,<5.0.0',
    
    # Cloud integration
    'boto3>=1.26.0,<2.0.0',
    'google-cloud-monitoring>=2.15.0,<3.0.0',
    'azure-monitor>=0.5.0,<1.0.0',
    
    # Message queues
    'pika>=1.3.0,<2.0.0',
    'kafka-python>=2.0.0,<3.0.0',
    
    # Testing and quality
    'pytest>=7.0.0,<8.0.0',
    'pytest-cov>=4.0.0,<5.0.0',
    'pytest-asyncio>=0.20.0,<1.0.0',
    'pytest-benchmark>=4.0.0,<5.0.0',
    'hypothesis>=6.0.0,<7.0.0',
    
    # Code quality
    'black>=23.0.0,<24.0.0',
    'flake8>=6.0.0,<7.0.0',
    'mypy>=1.0.0,<2.0.0',
    'isort>=5.12.0,<6.0.0',
    'bandit>=1.7.0,<2.0.0',
    
    # Documentation
    'sphinx>=6.0.0,<7.0.0',
    'sphinx-rtd-theme>=1.2.0,<2.0.0',
    'sphinx-autodoc-typehints>=1.22.0,<2.0.0',
    
    # Performance profiling
    'py-spy>=0.3.0,<1.0.0',
    'memory-profiler>=0.60.0,<1.0.0',
    'line-profiler>=4.0.0,<5.0.0',
    
    # Security
    'safety>=2.0.0,<3.0.0',
    'semgrep>=1.0.0,<2.0.0',
    
    # Deployment and orchestration
    'docker>=6.0.0,<7.0.0',
    'kubernetes>=26.0.0,<27.0.0',
    'ansible>=8.0.0,<9.0.0',
    'fabric>=3.0.0,<4.0.0',
    
    # Monitoring exporters
    'node-exporter>=0.1.0',
    'process-exporter>=0.1.0',
    
    # Additional utilities
    'click>=8.0.0,<9.0.0',
    'rich>=13.0.0,<14.0.0',
    'typer>=0.7.0,<1.0.0',
    'fire>=0.5.0,<1.0.0',
    'plotext>=5.0.0,<6.0.0',
]

# Platform-specific dependencies
platform_specific = {
    'linux': [
        'systemd-python>=234,<235',
        'dbus-python>=1.3.0,<2.0.0',
        'pygobject>=3.42.0,<4.0.0',
    ],
    'darwin': [
        'pyobjc>=9.0.0,<10.0.0',
    ],
    'win32': [
        'pywin32>=305,<306',
        'wmi>=1.5.0,<2.0.0',
    ],
}

# Add platform-specific dependencies
current_platform = sys.platform
if current_platform in platform_specific:
    install_requires.extend(platform_specific[current_platform])

# Development dependencies (not installed by default)
extras_require = {
    'dev': [
        # Development tools
        'pre-commit>=3.0.0,<4.0.0',
        'ipython>=8.0.0,<9.0.0',
        'jupyter>=1.0.0,<2.0.0',
        'jupyterlab>=4.0.0,<5.0.0',
        
        # Testing
        'pytest-xdist>=3.0.0,<4.0.0',
        'pytest-timeout>=2.1.0,<3.0.0',
        'pytest-mock>=3.10.0,<4.0.0',
        'pytest-html>=3.2.0,<4.0.0',
        'pytest-rerunfailures>=11.0.0,<12.0.0',
        
        # Documentation
        'sphinx-autobuild>=2021.0',
        'sphinxcontrib-mermaid>=0.8.0,<1.0.0',
        'sphinxcontrib-plantuml>=0.25,<0.26',
        
        # Code analysis
        'radon>=5.0.0,<6.0.0',
        'vulture>=2.0,<3.0',
        'pylint>=3.0.0,<4.0.0',
        'pytype>=2023.0.0',
        
        # Type checking
        'types-requests>=2.28.0',
        'types-PyYAML>=6.0.0',
        'types-python-dateutil>=2.8.0',
        'types-psutil>=5.9.0',
        
        # Coverage
        'coverage>=7.0.0,<8.0.0',
        'coveralls>=4.0.0,<5.0.0',
        
        # Build tools
        'build>=0.10.0,<1.0.0',
        'twine>=4.0.0,<5.0.0',
        'wheel>=0.40.0,<0.41.0',
        
        # Dependency management
        'pip-tools>=6.12.0,<7.0.0',
        'pip-audit>=2.5.0,<3.0.0',
        'dephell>=0.8.0,<0.9.0',
    ],
    'test': [
        # Core testing dependencies
        'pytest>=7.0.0,<8.0.0',
        'pytest-cov>=4.0.0,<5.0.0',
        'pytest-asyncio>=0.20.0,<1.0.0',
        'pytest-benchmark>=4.0.0,<5.0.0',
        'hypothesis>=6.0.0,<7.0.0',
        
        # Test utilities
        'factory-boy>=3.2.0,<4.0.0',
        'faker>=18.0.0,<19.0.0',
        'freezegun>=1.2.0,<2.0.0',
        'responses>=0.22.0,<0.23.0',
        'httmock>=1.4.0,<2.0.0',
        
        # Mocking
        'mock>=5.0.0,<6.0.0',
        'moto>=4.0.0,<5.0.0',
        
        # Test containers
        'testcontainers>=3.7.0,<4.0.0',
        'docker>=6.0.0,<7.0.0',
    ],
    'docs': [
        # Documentation
        'sphinx>=6.0.0,<7.0.0',
        'sphinx-rtd-theme>=1.2.0,<2.0.0',
        'sphinx-autodoc-typehints>=1.22.0,<2.0.0',
        'sphinx-autobuild>=2021.0',
        'sphinxcontrib-mermaid>=0.8.0,<1.0.0',
        'sphinxcontrib-plantuml>=0.25,<0.26',
        'myst-parser>=1.0.0,<2.0.0',
        'nbsphinx>=0.9.0,<1.0.0',
        'ipython>=8.0.0,<9.0.0',
    ],
    'performance': [
        # Performance monitoring
        'py-spy>=0.3.0,<1.0.0',
        'memory-profiler>=0.60.0,<1.0.0',
        'line-profiler>=4.0.0,<5.0.0',
        'filprofiler>=2022.0',
        'scalene>=1.5.0,<2.0.0',
        
        # Benchmarking
        'pytest-benchmark>=4.0.0,<5.0.0',
        'locust>=2.15.0,<3.0.0',
        'wrk>=0.1.0',
        
        # Performance analysis
        'snakeviz>=2.1.0,<3.0.0',
        'gprof2dot>=2022.0',
        'pyinstrument>=4.0.0,<5.0.0',
    ],
    'security': [
        # Security scanning
        'safety>=2.0.0,<3.0.0',
        'semgrep>=1.0.0,<2.0.0',
        'bandit>=1.7.0,<2.0.0',
        'trufflehog>=3.0.0,<4.0.0',
        
        # Dependency security
        'pip-audit>=2.5.0,<3.0.0',
        'ossaudit>=0.1.0',
        
        # Cryptography
        'cryptography>=38.0.0,<42.0.0',
        'pyjwt>=2.6.0,<3.0.0',
        'bcrypt>=4.0.0,<5.0.0',
    ],
    'ml': [
        # Machine learning
        'tensorflow>=2.10.0',
        'torch>=1.13.0',
        'xgboost>=1.7.0',
        'lightgbm>=3.3.0',
        'catboost>=1.0.0',
        
        # Deep learning
        'transformers>=4.26.0,<5.0.0',
        'datasets>=2.10.0,<3.0.0',
        'torchvision>=0.14.0,<1.0.0',
        
        # ML monitoring
        'mlflow>=2.0.0,<3.0.0',
        'wandb>=0.15.0,<1.0.0',
        'comet-ml>=3.0.0,<4.0.0',
        
        # Feature store
        'feast>=0.28.0,<1.0.0',
        'hopsworks>=3.0.0,<4.0.0',
    ],
    'monitoring': [
        # Monitoring systems
        'grafana-api>=1.0.0,<2.0.0',
        'prometheus-api-client>=0.5.0,<0.6.0',
        'influxdb-client>=1.30.0,<2.0.0',
        'elasticsearch>=8.0.0,<9.0.0',
        
        # Alerting
        'alertmanager-client>=0.1.0',
        'pagerduty-api>=0.1.0',
        'slack-sdk>=3.0.0,<4.0.0',
        
        # Dashboard
        'dash>=2.9.0,<3.0.0',
        'dash-bootstrap-components>=1.4.0,<2.0.0',
        'plotly>=5.10.0,<6.0.0',
        
        # Metrics collection
        'node-exporter>=0.1.0',
        'process-exporter>=0.1.0',
        'cadvisor-exporter>=0.1.0',
    ],
    'cloud': [
        # Cloud providers
        'boto3>=1.26.0,<2.0.0',
        'google-cloud-monitoring>=2.15.0,<3.0.0',
        'azure-monitor>=0.5.0,<1.0.0',
        'digitalocean>=1.0.0,<2.0.0',
        
        # Cloud native
        'kubernetes>=26.0.0,<27.0.0',
        'openshift>=0.13.0,<1.0.0',
        'helm>=0.1.0',
        
        # Infrastructure as code
        'terraform>=0.1.0',
        'pulumi>=3.0.0,<4.0.0',
        'ansible>=8.0.0,<9.0.0',
        
        # Container orchestration
        'docker>=6.0.0,<7.0.0',
        'docker-compose>=1.29.0,<2.0.0',
        'podman>=0.1.0',
    ],
    'all': [
        # Include all extra dependencies
        *install_requires,
        *[dep for deps in extras_require.values() for dep in deps],
    ],
}

# Setup configuration
setup(
    name='system_monitor',
    version=get_version(),
    packages=['system_monitor', 'system_monitor.cli'],
    package_dir={'': 'src'},
    data_files=get_data_files(),
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    
    # Package metadata
    author='Robotics Engineering Team',
    author_email='robotics-team@company.com',
    maintainer='Robotics Engineering Team',
    maintainer_email='robotics-team@company.com',
    description='Production-grade system monitoring for robotic systems',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    license='Proprietary',
    
    # URLs
    url='https://internal.company.com/git/robot-system-monitor',
    project_urls={
        'Documentation': 'https://internal.company.com/docs/robot-monitoring',
        'Source': 'https://internal.company.com/git/robot-system-monitor',
        'Tracker': 'https://internal.company.com/jira/projects/ROBOT',
        'Changelog': 'https://internal.company.com/docs/robot-monitoring/changelog',
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'Intended Audience :: Manufacturing',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Monitoring',
        'Topic :: System :: Logging',
        'Topic :: System :: Systems Administration',
        'License :: Other/Proprietary License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Framework :: ROS2',
        'Framework :: FastAPI',
        'Framework :: Flask',
        'Topic :: Database',
        'Topic :: Internet :: Log Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: Software Development :: Testing',
        'Topic :: System :: Benchmark',
        'Topic :: System :: Clustering',
        'Topic :: System :: Distributed Computing',
        'Topic :: System :: Hardware',
        'Topic :: System :: Networking :: Monitoring',
        'Topic :: System :: Recovery Tools',
        'Topic :: System :: Shells',
        'Topic :: Utilities',
    ],
    
    # Keywords
    keywords=[
        'ROS2', 'robotics', 'monitoring', 'performance', 'logging', 'system',
        'health', 'metrics', 'analytics', 'dashboard', 'alerting', 'observability',
        'production', 'enterprise', 'industrial', 'manufacturing', 'automation',
        'machine-learning', 'ai', 'computer-vision', 'iot', 'edge-computing',
        'container', 'kubernetes', 'docker', 'cloud', 'microservices',
        'devops', 'mlops', 'dataops', 'gitops', 'ci-cd',
    ],
    
    # Python requirements
    python_requires='>=3.8, <3.13',
    
    # Package discovery
    package_data={
        'system_monitor': [
            '*.py',
            '*.yaml',
            '*.yml',
            '*.json',
            '*.md',
            '*.rst',
            'VERSION',
            'requirements.txt',
        ],
    },
    
    # Include non-Python files
    include_package_data=True,
    
    # Zip safe
    zip_safe=False,
    
    # Platform compatibility
    platforms=[
        'Linux',
        'MacOS X',
        'Windows',
    ],
    
    # Scripts to install
    scripts=[
        'scripts/start_monitoring.sh',
        'scripts/stop_monitoring.sh',
        'scripts/health_check.sh',
        'scripts/performance_report.sh',
        'scripts/log_analyzer.sh',
        'scripts/alert_manager.sh',
        'scripts/backup_monitor.sh',
        'scripts/restore_monitor.sh',
        'scripts/update_monitor.sh',
        'scripts/diagnose_monitor.sh',
        'scripts/benchmark_monitor.sh',
        'scripts/profile_monitor.sh',
    ],
    
    # Command line interface
    cmdclass={},
    
    # Test suite
    test_suite='pytest',
    tests_require=['pytest>=7.0.0'],
    
    # Dependency links (deprecated, using pip directly)
    dependency_links=[],
    
    # Obsoletes other packages
    obsoletes=[
        'robot_monitor',
        'ros2_system_monitor',
        'industrial_monitor',
    ],
    
    # Provides other packages
    provides=[
        'system_monitor',
        'robot_monitoring',
        'ros_monitoring',
    ],
    
    # Additional metadata for PyPI
    options={
        'bdist_wheel': {
            'universal': False,
        },
        'egg_info': {
            'tag_build': '',
            'tag_date': False,
        },
    },
    
    # Custom setup commands
    setup_requires=[
        'setuptools>=58.2.0',
        'wheel>=0.40.0',
    ],
    
    # Custom build configuration
    command_options={
        'build_sphinx': {
            'source_dir': ('setup.py', 'docs'),
            'build_dir': ('setup.py', 'docs/_build'),
            'fresh_env': ('setup.py', True),
            'all_files': ('setup.py', True),
        },
    },
    
    # Security considerations
    # This package may require system-level permissions for full functionality
    # (e.g., reading system metrics, managing services)
    
    # Privacy considerations
    # This package collects system metrics which may include sensitive information
    # Users should review and configure privacy settings appropriately
    
    # Performance considerations
    # The monitoring system itself consumes resources
    # Resource limits and sampling rates should be configured appropriately
    
    # Logging considerations
    # Log files can grow large, rotation and retention policies should be configured
    
    # Network considerations
    # Some features require network access for remote monitoring and alerting
    
    # Storage considerations
    # Metric data requires storage, retention policies should be configured
    
    # Backup considerations
    # Configuration and data should be backed up regularly
)

# Post-installation message
print("\n" + "="*70)
print("System Monitor Installation Complete")
print("="*70)
print("\nQuick Start:")
print("1. Source ROS2 environment: source /opt/ros/humble/setup.bash")
print("2. Launch monitoring system: ros2 launch system_monitor monitoring.launch.py")
print("3. Check status: monitor-status")
print("4. View metrics: monitor-metrics")
print("\nConfiguration:")
print("- Edit config files in: /etc/robot_monitor/")
print("- View logs in: /var/log/robot_monitor/")
print("- Performance data in: /var/performance_data/")
print("\nDocumentation:")
print("- User manual: https://internal.company.com/docs/robot-monitoring")
print("- API reference: https://internal.company.com/docs/robot-monitoring/api")
print("\nSupport:")
print("- Email: robotics-support@company.com")
print("- Slack: #robotics-support")
print("- Jira: ROBOT project")
print("="*70)
