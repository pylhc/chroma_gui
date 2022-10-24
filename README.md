# Non-Linear Chromaticity GUI

The Chromaticity GUI is a tool to compute non-linear chromaticity via
measurements done in the CCC.

# Deployment

Change the version in [__init__.py](./chroma_gui/__init__.py)

```bash
acc-py app lock .
acc-py app deploy .
acc-py app promote chroma-gui <version>
```
