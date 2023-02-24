import setuptools

setuptools.setup(
    name="ACEHAL",
    version="0.0.1",
    packages=setuptools.find_packages(exclude=["tests"]),
    install_requires=["ase", "julia", "click>=8.0", "numpy", "scipy", "timeout_decorator", "optuna"],
    # entry_points="""
    # [console_scripts]
    # tsa=trajstruct.cli:cli
    # """
)
