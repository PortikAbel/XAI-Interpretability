from setuptools import find_packages, setup

setup(
    name="xai-interpretability",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    license="MIT",
    author="UBB",
    author_email="your@email.com",
    description="Your sub-project",
)
