git clone https://github.com/mlcommons/medperf.git
cd medperf
pip install --force-reinstall -e ./cli
rm ~/mlcube.yaml && rm -rf ~/.medperf
medperf --version
medperf profile set --loglevel=debug
medperf auth synapse_login