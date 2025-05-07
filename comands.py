
'''

python3 -m venv .venv
conda create -n env-01 python=3.9 numpy pandas
conda activate env-01
python --version


pip install kedro
pip install kedro-viz

kedro info

kedro new --starter=spaceflights-pandas

kedro ipython
shuttles = catalog.load("shuttles")
shuttles.head()

OUTSIDE KEDRO IPYTHON 
kedro run --nodes=preprocess_companies_node
kedro run
kedro run --nodes=preprocess_companies_node,preprocess_shuttles_node
kedro run --pipeline=data_science

kedro viz run
ctrl+c


# IR A LA CARPETA DEL PROYECTO KEDRO PRIMERO.
kedro jupyter notebook




kedro info
python -V
pip install black # REFORMATING
black ambs-analytics-project/src/ambs_analytics_project/
kedro registry list
kedro run --pipeline data_processing
kedro viz
kedro pipeline create data_science


SMOTE AND RUS
  Desicion tree, MLP clasifier.

'''


'''

'''


'''


'''
