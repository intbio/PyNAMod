import setuptools

setuptools.setup(
    name="pynamod",
    version="0.1",
    python_requires="==3.13",
    packages=setuptools.find_packages(exclude=['examples']),
    data_files=['pynamod/atomic_analysis/classifier.pkl', 'pynamod/tests/cg_3lz0.h5'],
    include_package_data=True
)
