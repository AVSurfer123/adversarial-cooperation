import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="acrl", # Replace with your own username
    version="0.0.1",
    author="Ashwin Vangipuram and Shawn Shacterman",
    author_email="author@example.com",
    description="CS 285 adverserial cooperation final project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AVSurfer123/adverserial-cooperation",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
