# Evaluation packages
pip install -r requirements.txt
cd bleurt
pip install .
cd ..

# Architectures packages
cd transformers
pip install .
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
cd ..

# Download bleurt checkpoint
cd scripts/benchmarking
wget https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip .
unzip bleurt-large-512.zip
cd ../..