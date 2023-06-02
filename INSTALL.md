# Installation instructions

1. Install anaconda distribution of Python 3.10 https://docs.anaconda.com/anaconda/install/index.html
2. In GitHub, go to https://github.com/settings/tokens
3. Generate a new token (classic) with a long expiration date (you can always make another if need be)
4. Under "Select Scopes" check `repo`, `workflow`, `write:packages`
5. Click the green "Create Personal Access Token" button and save the number it gives you
6. Create an empty local directory and navigate to it `mkdir CIEMSS; cd CIEMSS`
7. Clone pyciemss repository: `git clone https://github.com/ciemss/pyciemss.git`. Use your GitHub username, then as password use the Personal Access Token you kept (should only be needed the first time)
8. Navigate into the cloned folder `cd pyciemss`
9. Create and activate a conda virtual environment `conda create -n CIEMSS_ENV python=3.10; conda activate CIEMSS_ENV` (if prompted to Proceed ([y]/n)? type y and enter)
10. Install dependencies. `pip install -e .` (again, y if prompted)
11. Setup `CIEMSS_ENV` Jupyter kernel `python -m ipykernel install --user --name CIEMSS_ENV --display-name "Python CIEMSS"`
12. Initiate a Jupyter environment. `jupyter notebook` or open chosen IDE like VSCode
13. Open `demo/scenario2.ipynb.`
14. If in IDE, change kernel to either the kernel we set before `python CIEMSS`, the python environment `CIEMSS_ENV`, or an active jupyter server through its given URL
15. Run the notebook!
