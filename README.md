# Pyciemss



## Getting started

To make it easy for you to get started with the pyciemss repository, here's a list of recommended steps:

1. install `pyenv` and `pyenv-virtualenv`:

```bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cd ~/.pyenv && src/configure && make -C src
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
pyenv install 3.10.9
```

2. Install pyciemss and set up a branch-specific virtual environment

```bash
git clone https://github.com/ciemss/pyciemss.git <branch-name>
cd <branch-name>
git checkout <branch-name>
pyenv virtualenv 3.10.9 <branch-name>
pyenv shell <branch-name>
pip install -e causal_pyro  # until causal_pyro is publicly released
pip install -e .
```

Now you are good to go.