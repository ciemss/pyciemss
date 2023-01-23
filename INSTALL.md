# Installation instructions

1. Create an empty directory and navigate to it `mkdir CIEMSS; cd CIEMSS`
2. In GitLab, go to https://gitlab.pnnl.gov/-/profile/personal_access_tokens
3. Set the Expiration Date a year from now minus one day
4. Under "Select Scopes" check read_repository and write_repository
5. Click the green "Create Personal Access Token" button and save the number it gives you
6. Clone `causal_pyro` repository `git clone https://gitlab.pnnl.gov/ciemss/causal_pyro.git`
7. use your gitlab username, then as password use the Personal Access Token you kept
8. Clone pyciemss repository: `git clone https://gitlab.pnnl.gov/ciemss/pyciemss.git`
(you should only need to login once, but just in case) For username <gitlab username>, password the Personal Access Token
9. Install anaconda distribution of Python 3.10 https://docs.anaconda.com/anaconda/install/index.html
10. Create and activate a conda virtual environment `conda create -n CIEMSS_ENV python=3.10; conda activate CIEMSS_ENV` (if prompted to Proceed ([y]/n)? type y and enter)
11. Install causal pyro dependencies. `pip install -e causal_pyro/`
12. Install pyciemss dependencies. `pip install -r pyciemss/requirements.txt`
13. Setup `CIEMSS_ENV` Jupyter kernel `python -m ipykernel install --user --name CIEMSS_ENV --display-name "Python CIEMSS"`
14. Initiate a Jupyter environmentjupyter notebook
15. Open `demo/scenario2.ipynb.`
16. Use the `CIEMSS_ENV` environment we created earlier.
17. On the drop downs navigate using `“Kernel” -> “Change Kernel” -> “Python CIEMSS”`
18. Run the notebook!