from setuptools import setup, find_packages

reqs=[
    ]

setup(
    name='MiDi',
    version='0.0.1',
    url=None,
    author='Clement Vignac, Nagham Osman',
    author_email='clement.vignac@epfl.ch',
    description='MiDi: Mixed Graph and 3D Denoising Diffusion for Molecule Generation',
    packages=find_packages(exclude=["wandb", "archives", "configs"]),
    install_requires=reqs
)
