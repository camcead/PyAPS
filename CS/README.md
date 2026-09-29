# WEAVE Contribution Software Installation Guide

This document provides step-by-step instructions to install and configure the contribution software packages used within the WEAVE project.

---

## 1. AMY

```bash
git clone git@github.com:mmonguio/AMY.git
cd AMY
unzip ptemcee-1.0.0.zip
python3 -m pip install -e .
```

---

## 2. FESWI

```bash
git clone git@gitlab.com:daguado/feswi.git FESWI
```
---

## 3. RRLEW and RRLGV

```bash
git clone git@github.com:NikolayBritavskiyAstro/CS_RRLEW.git RRLEW
git clone git@github.com:NikolayBritavskiy/CS_RRLGV.git RRLGV
```

No installation required.

---

## 4. SPACE

### Dependencies

On **Ubuntu**:
```bash
sudo apt-get update
sudo apt-get install -y liblapack-dev liblapack3 libopenblas-base libopenblas-dev libatlas-base-dev
```

On **Mac**:  
Refer to this guide → [Installing LAPACK and BLAS on macOS](https://pheiter.wordpress.com/2012/09/04/howto-installing-lapack-and-blas-on-mac-os/)

### Installation

```bash
git clone git@github.com:corrado-github/SPAce_WEAVE.git SPACE
cd SPACE

# Edit the Makefile to set the correct Fortran compiler
make clean
make
make install   # optional (ignore failures)
```

📖 Tutorial: [Touttarila documentation](https://dc.g-vo.org/sp_ace/q/dist/static/tutorial.pdf)

---

## 5. SQUEzE

```bash
git clone https://github.com/iprafols/SQUEzE.git
cd SQUEzE
python3 setup.py develop

# Run installation test
python3 setup.py test

# Create training set
cd ../bin
python3 pretrain_squeze.py
```

---

## 6. ALFA & NEAT

```bash
mkdir ALFA_NEAT
cd ALFA_NEAT

git clone git@github.com:rwesson/ALFA.git
git clone git@github.com:rwesson/NEAT.git
```

### Makefile adjustment

In both `ALFA/Makefile` and `NEAT/Makefile`, add:

```make
PREFIX=$(HOME)/.local
DESTDIR=""
```

### Build & Install

```bash
cd ALFA   # or cd NEAT
make
make install   # required, installs necessary libraries
```

### Uninstall

```bash
make clean
make uninstall
```

---

## 7. SAPP

```bash
git clone git@github.com:JGerbs13/SAPP-WEAVE.git
```

---

## Notes

- Always ensure you are using the correct compiler and Python environment (`python3` recommended).
- Some packages require system libraries (e.g., LAPACK, BLAS, OpenBLAS).
- Use `virtualenv` or `conda` environments where possible to isolate dependencies.

---
