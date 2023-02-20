import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import platform
from glob import glob
import subprocess

git_home = subprocess.check_output(
    ['git', 'rev-parse', '--show-toplevel'], text=True).strip('\n')

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
    if len(notebooks_list) == 0:
        raise Exception("No notebooks found!")
    # Some of the notebooks need data files which are not pushed to github due
    # to large file size. We skip these notebooks from testing.
    no_test_list = [f"{git_home}/examples/load_waveform_demo.ipynb"]
    for notebook in notebooks_list:
        if notebook in no_test_list:
            # skip these notebooks, otherwise it will fail due to missing data
            # file.
            print(f"No test for {notebook}")
            continue
        print(f'testing {notebook}')
        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=None, kernel_name=python_version)
        ep.preprocess(nb, {'metadata': {'path': '.'}})
