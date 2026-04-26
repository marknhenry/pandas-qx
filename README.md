
# pandas-qx

A small pandas extension that adds finance-oriented DataFrame methods
using pandas accessors.

# Steps to Update
pip install build twine
python -m build
twine upload dist/*


## Installation
```bash
pip install pandas-qx
```

In pandas 
```python
%load_ext autoreload
%autoreload 2
```

# Work with editable version of package
```bash
pip install .e
```

# Releasing a New Version to PyPI (`pandas-qx`)

This document describes the **manual release process** for publishing a new version of `pandas-qx` to PyPI.

---

## 1. Update the Version Number

Edit `pyproject.toml` and bump the version following semantic versioning.

```toml
[project]
name = "pandas-qx"
# pandas_qx/__init__.py
__version__ = "0.4.0"
```

## 2. Update the Changelog in CHANGELOG.md

```
## 0.4.0 – 26 Apr 2026

### Added
- New XYZ feature

### Fixed
- Bug in ABC
```

## 3. Commit the Release Changes
```bash
git add .
git commit -m "Release 0.x.x"
git tag v0.x.x
git push origin main
git push origin v0.x.x
```

## 4. Build the distribution artifacts & Upload
```bash
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

## 5. Verify
```bash
pip install --upgrade pandas-qx
python - <<EOF
import pandas_qx
print(pandas_qx.__version__)
EOF
```