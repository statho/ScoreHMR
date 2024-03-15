from setuptools import setup, find_packages

print("Found packages:", find_packages())
setup(
    description="ScoreHMR as a package",
    name="score_hmr",
    packages=find_packages(),
)
