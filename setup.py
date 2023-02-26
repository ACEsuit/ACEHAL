import setuptools

setuptools.setup(
    name="ACEHAL",
    version="0.0.1",
    packages=setuptools.find_packages(exclude=["tests"]),
    install_requires=["ase", "julia", "numpy", "matplotlib", "scipy", "scikit-learn", "timeout_decorator", "optuna", "click>=8.0"],
    # entry_points="""
    # [console_scripts]
    # tsa=trajstruct.cli:cli
    # """
)
