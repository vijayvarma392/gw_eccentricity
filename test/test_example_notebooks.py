import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import platform
from glob import glob
import os, sys, subprocess

git_home = subprocess.check_output(['git', 'rev-parse',
    '--show-toplevel'], text=True).strip('\n') 

python_version = platform.python_version()
if python_version[:2] == '2.':
    python_version = 'python2'
elif python_version[:2] == '3.':
    python_version = 'python3'

def test_example_notebooks():
    """ Tests that all the example ipython notebooks of format
    gw_eccentricity/examples/*.ipynb are working. Since we expect these to be
    used by our users, it would be emabarassing if our own examples failed.
    """
    notebooks_list = glob(f'{git_home}/examples/*.ipynb')
    notebooks_list.sort()
    # change dir to get the correct path to data files
    curret_dir = os.getcwd()
    os.chdir(f"{git_home}/examples/")

    if len(notebooks_list) == 0:
        raise Exception("No notebooks found!")

    for notebook in notebooks_list:

        print(f'testing {notebook}')
        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=None, kernel_name=python_version)
        ep.preprocess(nb, {'metadata': {'path': '.'}})

    # change back dir
    os.chdir(curret_dir)
