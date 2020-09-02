# Architectures packages
cd transformers
pip install .
pip install -r examples/requirements.txt
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
cd ..

# Evaluation packages