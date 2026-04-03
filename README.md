# SVM Classifier

A Support Vector Machine (SVM) classifier built from scratch in C++ using the SMO (Sequential Minimal Optimization) algorithm. No external libraries and all components implemented manually.

## Features

- SVM with SMO training algorithm
- Three kernel functions: Linear, RBF, Polynomial
- Z-score feature normalisation
- Evaluation: accuracy, confusion matrix, precision, recall, F1 score
- 5-fold cross-validation
- Kernel comparison

## Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) — 569 samples, 30 features, binary classification (Malignant / Benign).

Download the dataset and place it at `data/wdbc.csv`:
```bash
mkdir -p data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" -o data/wdbc.csv
```

## Build and Run

Requires CMake and a C++17 compiler.

```bash
cmake -B build
cmake --build build
./build/svm_classifier
```

## Project Structure

```
src/
  main.cpp           — Entry point
  data_loader.hpp/cpp — CSV parsing, normalisation, train/test split
  kernel.hpp/cpp      — Linear, RBF, Polynomial kernels
  svm.hpp/cpp         — SVM class with SMO solver
  evaluation.hpp/cpp  — Metrics and k-fold cross-validation
```
