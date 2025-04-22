import setuptools
setuptools.setup(     
     name="pynamod",     
     version="0.1",
     python_requires=">=3.11",   
     packages=setuptools.find_packages(exclude=['examples']),
     data_files=['pynamod/atomic_analysis/classifier.pkl'],
     include_package_data=True
)