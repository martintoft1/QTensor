import setuptools

# Configure the required packages and scripts to install.
REQUIRED_PACKAGES = [
    'numpy>=1.18.1',
    'networkx>=2.3',
    'matplotlib>=3.1.3'
    ,'google-api-core[grpc]<=1.14.0'
    ,'cirq'
    ,'qiskit'
    ,'qiskit-optimization'
    ,'pyrofiler>=0.1.5'
    ,'loguru'
    ,'tqdm'
    ,'click'
    ,'qtensor-qtree'
    ,'lazy-import'
    ,'pynauty'
    ,'sarge'
    ,'mongocat'
    ,'PyInquirer'
    ,'acqdp==0.1.1'
    ,'GPUtil'
]

MORE = [
    'cartesian_explorer==0.1.13',
    'cirq==1.1.0',
    'cirq_core==1.1.0',
    'click==8.0.4',
    'cotengra==0.6.0',
    'cycler==0.11.0',
    'fire==0.6.0',
    'Flask==1.1.2',
    'flask_cors==4.0.1',
    'loguru==0.7.2',
    'matplotlib==3.5.1',
    'mongocat==0.2.1',
    'mpi4py==3.1.6',
    'networkx==2.7.1',
    'numpy==1.23.5',
    'pandas==1.4.2',
    'psutil==5.8.0',
    'pymongo==4.7.1',
    'pyrofiler==0.1.11',
    'pytest==7.1.1',
    'qiskit==0.45.1',
    'qiskit_terra==0.45.1',
    'quimb==1.8.1',
    'sarge==0.1.7.post1',
    'scipy==1.13.0',
    'seaborn==0.11.2',
    'setuptools==67.6.1',
    'tensornetwork==0.4.6',
    'tqdm==4.64.0',
]

setuptools.setup(name='qtensor',
                 version='0.1.2',
                 description='Framework for efficient quantum circuit simulations',
                 url='https://github.com/danlkv/qtensor',
                 keywords='quantum_circuit quantum_algorithms',
                 author='D. Lykov, et al.',
                 author_email='dan@qtensor.org',
                 license='Apache',
                 packages=setuptools.find_packages(),
                 install_requires=REQUIRED_PACKAGES,
                 setup_requires=[
                    'Cython',  
                ],
                 extras_require={
                     'tensorflow': ['tensorflow<=1.15'],
                 },
                 include_package_data=True,
                 zip_safe=False)
