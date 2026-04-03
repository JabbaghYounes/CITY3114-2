# SVM Classifier for Breast Cancer Diagnosis

**Module:** CITY3114 — Machine Learning and Algorithms
**Assignment:** 2
**Name:** [Your Name]
**Student ID:** [Your Student ID]

---

## Task 1: Implementation and Evaluation

### 1.1 Description of Program

This project implements a Support Vector Machine (SVM) classifier from scratch in C++, without relying on any external machine learning libraries. The classifier uses the Sequential Minimal Optimization (SMO) algorithm for training, which is the standard approach for solving the quadratic programming problem at the heart of SVM learning.

The program is built with a modular structure:

- **data_loader** — handles CSV parsing, feature normalisation, and splitting data into training and test sets.
- **kernel** — provides three kernel functions (Linear, RBF, Polynomial) behind a shared interface, allowing the SVM to switch between them easily.
- **svm** — contains the core SVM class with the SMO training loop, prediction logic, and support vector extraction.
- **evaluation** — computes performance metrics (accuracy, precision, recall, F1 score, confusion matrix) and implements k-fold cross-validation and grid search for hyperparameter tuning.
- **main** — ties everything together: loads the data, trains models, runs the grid search, and outputs results.

#### Dependencies and How to Run

The only requirements are a C++17 compiler (such as g++) and CMake (version 3.14 or above). No third-party libraries are needed.

To build and run:

```bash
cmake -B build
cmake --build build
./build/svm_classifier
```

The dataset file (`wdbc.csv`) must be placed in a `data/` directory at the project root. It can be downloaded from the UCI Machine Learning Repository:

```bash
mkdir -p data
curl -sL "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data" -o data/wdbc.csv
```

### 1.2 Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine Learning Repository (Wolberg et al., 1995). It contains 569 samples of fine needle aspirate (FNA) images of breast masses, where each sample is classified as either **Malignant (M)** or **Benign (B)**.

Each sample has 30 numerical features, computed from ten real-valued properties of the cell nuclei:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave points
9. Symmetry
10. Fractal dimension

For each of these ten properties, three statistics are recorded: the mean, the standard error, and the worst (largest) value — giving 30 features in total.

The class distribution is slightly imbalanced: 212 malignant cases (37.3%) and 357 benign cases (62.7%). This is worth noting because in a medical context, failing to detect a malignant tumour (a false negative) is far more costly than a false positive. This makes recall a particularly important metric alongside accuracy.

For preprocessing, the ID column was discarded, the diagnosis was mapped to numerical labels (M → +1, B → −1), and all 30 features were normalised using z-score standardisation (zero mean, unit variance). Feature normalisation is important for SVMs because the algorithm relies on distances between data points — without it, features with larger scales would dominate the decision boundary.

### 1.3 Algorithm Implementation

#### Support Vector Machines

A Support Vector Machine is a supervised learning algorithm that finds the optimal hyperplane separating two classes of data. The "optimal" hyperplane is the one that maximises the margin — the distance between the decision boundary and the nearest data points from each class. These nearest points are called support vectors, and they are the only samples that influence the position of the boundary.

The key strength of SVMs is the kernel trick. Rather than operating directly in the input space, kernel functions implicitly map data into a higher-dimensional feature space where a linear separator may exist, even if the original data is not linearly separable. Three kernel functions were implemented:

- **Linear:** K(a, b) = a · b — a simple dot product, effective when data is linearly separable.
- **RBF (Radial Basis Function):** K(a, b) = exp(−γ‖a − b‖²) — measures similarity based on distance, capable of handling complex non-linear boundaries. The parameter γ controls how quickly the similarity falls off with distance.
- **Polynomial:** K(a, b) = (γ · a · b + c₀)^d — maps data into a polynomial feature space, with degree d controlling the complexity.

#### SMO Algorithm

Training an SVM involves solving a constrained quadratic optimisation problem over the Lagrange multipliers (α values), one per training sample. The Sequential Minimal Optimization (SMO) algorithm, developed by John Platt (1998), breaks this large problem into the smallest possible sub-problems: at each step, it selects a pair of α values and optimises them jointly while holding all others fixed.

The implementation works as follows:

1. Initialise all α values to zero and the bias term b to zero.
2. Precompute the full kernel matrix K(i, j) for all pairs of training samples to avoid redundant calculations.
3. Maintain an error cache E(i) = f(i) − y(i), where f(i) is the current SVM output for sample i.
4. Iterate over all samples. For each sample i, check whether it violates the KKT (Karush-Kuhn-Tucker) conditions beyond a set tolerance.
5. If a violation is found, select a second sample j using the second-choice heuristic — picking the j that maximises |E(i) − E(j)| to make the largest possible optimisation step.
6. Compute the bounds L and H for α_j, the learning step η, and update both α_i and α_j.
7. Update the bias term b based on whether the new α values are on the boundary or not.
8. Recompute the error cache and repeat until no more KKT violations are found or the maximum number of iterations is reached.

After training, only samples with α > 0 are retained as support vectors. These are the only samples needed for making predictions, which makes the trained model compact.

#### Key Parameters

- **C (regularisation):** Controls the trade-off between maximising the margin and minimising classification errors. A small C allows a wider margin with some misclassifications; a large C penalises errors more heavily, potentially overfitting.
- **γ (gamma):** Used by the RBF and Polynomial kernels. A small γ gives each training sample broad influence, resulting in smoother boundaries. A large γ makes the model focus more on individual samples, which can lead to overfitting.
- **Degree:** Used by the Polynomial kernel. Higher degrees allow more complex boundaries but increase the risk of overfitting.
- **Tolerance:** The threshold for KKT violation checks (set to 0.001). Smaller values give more precise solutions but take longer to converge.
- **Max iterations:** An upper limit on SMO passes (set to 1000), acting as a safeguard against non-convergence.

### 1.4 Training

The dataset was split into 80% training (456 samples) and 20% test (113 samples) using a seeded random shuffle for reproducibility.

#### Initial Training

Each kernel was first trained with default parameters (C = 1.0, γ = 0.1, degree = 3) to establish a baseline. The initial results revealed that while RBF performed reasonably well, the Linear and Polynomial kernels performed poorly with these defaults — indicating that the choice of hyperparameters matters significantly.

#### Hyperparameter Optimisation

To find better parameters, a grid search was conducted over the following ranges:

- C: {0.1, 1, 10, 100}
- γ: {0.001, 0.01, 0.1, 1}
- Degree (Polynomial only): {2, 3, 4}

Each combination was evaluated using 5-fold cross-validation on the full normalised dataset. This means the data was split into 5 equal parts, and the model was trained on 4 parts and tested on the remaining part, rotating through all 5 combinations. The reported accuracy is the average across all folds.

This approach was chosen because it gives a more reliable estimate of how well a model will generalise to unseen data, compared to evaluating on a single train/test split. It also helps avoid selecting hyperparameters that happen to work well on one particular split but poorly on others.

The grid search identified the following optimal parameters:

| Kernel | Best C | Best γ | Best Degree | CV Accuracy |
|---|---|---|---|---|
| Linear | 0.1 | — | — | 72.41% |
| RBF | 0.1 | 0.01 | — | 90.16% |
| Polynomial | 1.0 | 0.001 | 3 | 89.41% |

### 1.5 Results

#### Default Parameters (C = 1.0, γ = 0.1)

| Kernel | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Linear | 41.59% | 0.3945 | 1.0000 | 0.5658 |
| RBF | 84.07% | 0.9630 | 0.6047 | 0.7429 |
| Polynomial | 69.03% | 1.0000 | 0.1860 | 0.3137 |

With default parameters, only the RBF kernel produced usable results. The Linear kernel predicted almost everything as malignant (high recall but very low precision), while the Polynomial kernel predicted almost everything as benign (high precision but very low recall). This demonstrates why hyperparameter tuning is essential.

#### Optimised Parameters

| Kernel | Accuracy | Precision | Recall | F1 Score | Support Vectors |
|---|---|---|---|---|---|
| Linear | 97.35% | 1.0000 | 0.9302 | 0.9639 | 16 |
| RBF | 90.27% | 1.0000 | 0.7442 | 0.8533 | 18 |
| Polynomial | 95.58% | 0.9318 | 0.9535 | 0.9425 | 24 |

After optimisation, all three kernels improved substantially. The most striking result is the **Linear kernel achieving 97.35% accuracy** — the highest of the three. This suggests that the Breast Cancer Wisconsin dataset, once normalised, is largely linearly separable. The Linear kernel also had the fewest support vectors (16), meaning it built the simplest model.

The RBF kernel, despite being the most flexible, achieved the lowest accuracy of the three after tuning. Its recall of 0.74 means it still missed about a quarter of malignant cases, which is a concern for a medical application.

The Polynomial kernel with degree 3 achieved a good balance of precision and recall, reflected in its strong F1 score of 0.9425.

#### Confusion Matrix (Best Model — Linear, Optimised)

|  | Predicted Malignant (+1) | Predicted Benign (−1) |
|---|---|---|
| **Actual Malignant** | 40 | 3 |
| **Actual Benign** | 0 | 70 |

The optimised Linear SVM correctly classified 110 out of 113 test samples. It produced zero false positives and only 3 false negatives. In a clinical setting, those 3 missed malignant cases would still be a concern, but the overall performance is strong for a from-scratch implementation.

---

## Task 2: Discussion

### 2.1 Ability of the SVM Algorithm

SVMs are well suited to a range of classification problems, particularly those involving high-dimensional data. The Breast Cancer Wisconsin dataset, with 30 features and 569 samples, is a good example — SVMs handle cases where the number of features is large relative to the number of samples better than many other algorithms.

The kernel trick is central to this versatility. By swapping in different kernel functions, the same SVM framework can model linear boundaries, smooth non-linear boundaries (RBF), or polynomial decision surfaces. This was demonstrated in this project: the same SMO training code produced very different classifiers simply by changing the kernel.

SVMs also generalise well to unseen data when properly tuned. The margin-maximisation objective acts as a form of regularisation — by seeking the widest possible margin, SVMs naturally resist overfitting. The regularisation parameter C provides additional control, allowing the user to balance margin width against training accuracy.

Another practical advantage is that after training, only the support vectors are needed for prediction. In this project, the optimised Linear SVM used just 16 support vectors out of 456 training samples, making predictions efficient.

SVMs work well for binary classification tasks such as spam detection, image classification (e.g., face detection), handwriting recognition, and medical diagnosis — any problem where a clear boundary between two classes exists or can be found through a kernel mapping.

### 2.2 Limitations of the SVM Algorithm

Despite their strengths, SVMs have several practical limitations.

**Scalability:** The SMO algorithm requires computing kernel values between pairs of training samples. In this implementation, the full kernel matrix is precomputed, which uses O(n²) memory. For the 569-sample dataset this is fine, but for datasets with tens of thousands of samples or more, this becomes impractical. Training time also scales poorly — roughly O(n² × iterations) — which makes SVMs a poor choice for very large datasets.

**Sensitivity to hyperparameters:** As the results showed, the difference between default and optimised parameters was dramatic (Linear went from 42% to 97%). SVMs do not work well "out of the box" — finding good values of C, γ, and degree requires systematic search, which itself is computationally expensive.

**Multi-class classification:** SVMs are inherently binary classifiers. Extending them to problems with more than two classes requires strategies like one-vs-one or one-vs-rest, which add complexity and can be inefficient for large numbers of classes.

**Interpretability:** Unlike decision trees or logistic regression, SVMs do not produce easily interpretable models. The decision boundary is defined implicitly through support vectors and kernel evaluations, making it difficult to explain why a particular prediction was made. In medical settings, where clinicians need to understand and trust model decisions, this lack of transparency can be a significant drawback.

**Feature scaling:** SVMs are sensitive to the scale of input features. Without normalisation, features with larger ranges dominate the distance calculations. This was addressed in this project using z-score normalisation, but it adds a preprocessing step that must be applied consistently to both training and test data.

### 2.3 Real-World Applications

SVMs are used across a wide range of fields today:

- **Medical diagnosis:** Beyond breast cancer classification, SVMs have been applied to detecting skin cancer from dermoscopy images, predicting heart disease risk from patient records, and classifying brain tumours from MRI scans. Their ability to work with high-dimensional feature spaces makes them well suited to genomic and proteomic data analysis.

- **Text classification:** SVMs were among the first algorithms to achieve strong results in spam filtering, sentiment analysis, and document categorisation. In natural language processing, text is typically represented as high-dimensional sparse vectors (bag-of-words or TF-IDF), which is a setting where SVMs perform particularly well.

- **Image recognition:** Before deep learning became dominant, SVMs were widely used for tasks like handwritten digit recognition (e.g., postal code reading), face detection, and object classification. They remain useful in scenarios where training data is limited, since they can generalise from relatively few examples.

- **Bioinformatics:** SVMs are commonly used for protein classification, gene expression analysis, and predicting protein-protein interactions. These problems often involve datasets with many features and relatively few samples — exactly the kind of problem SVMs handle well.

- **Finance:** Applications include credit scoring, fraud detection, and stock market prediction. SVMs can model non-linear relationships in financial data while the regularisation parameter helps prevent overfitting to noisy market signals.

While deep learning has replaced SVMs in many large-scale applications, SVMs remain a practical choice when data is limited, interpretability requirements are moderate, and the problem is well suited to kernel methods. They continue to see active use in scientific research and specialised domains where their mathematical foundations and strong generalisation properties are valued.

---

## References

1. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic) Data Set*. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

2. Platt, J. C. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*. Microsoft Research Technical Report MSR-TR-98-14.

3. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

4. Scikit-learn developers. *Support Vector Machines*. https://scikit-learn.org/stable/modules/svm.html

5. American Cancer Society. *How Common Is Breast Cancer?* https://www.cancer.org/cancer/types/breast-cancer/about/how-common-is-breast-cancer.html
