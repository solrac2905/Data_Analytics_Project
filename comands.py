
'''

python3 -m venv .venv
conda create -n env-01 python=3.9 numpy pandas
conda activate env-01
python --version


pip install kedro
kedro info

kedro new --starter=spaceflights-pandas

kedro ipython


'''