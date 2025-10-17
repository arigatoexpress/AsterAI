conda activate pytorch_build
conda install -y numpy mkl-devel ninja
pip install cmake typing_extensions pyyaml setuptools wheel requests
echo "Dependency installation complete."
