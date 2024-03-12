# Non-Linear Chromaticity GUI

The Chromaticity GUI is a tool to compute non-linear chromaticity via
measurements done in the CCC.

# Running

Be sure to have the `/acc` directory mounted, which can be done via:

```bash
sshfs cs-ccr-dev2:/acc/ /acc'
```

Running the GUI is then very simple thanks to acc-py.

```bash
source /acc/local/share/python/acc-py/base/pro/setup.sh
acc-py app run chroma-gui
```

# Deployment

* Change the version in [__init__.py](./chroma_gui/__init__.py)
* Update the [CHANGELOG](./CHANGELOG.md)

```bash
alias acc-py="/acc/local/share/python/acc-py/apps/acc-py-cli/pro/bin/acc-py"
acc-py app lock .
acc-py app deploy .
acc-py app promote chroma-gui <version>
```
