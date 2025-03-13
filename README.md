<div align="right">
  <img src="./assets/logo.png" alt="Logo" width="150">
</div>

# Classification Analysis of Vertebral Column Data

## Overview
This project involves analyzing the **vertebral column dataset** using **both unsupervised and supervised classification methods**. The aim is to explore the data, apply clustering and classification techniques, and interpret the results to gain insights into biomechanical attributes related to normal and abnormal spinal conditions.

## Dataset
The dataset used in this project is **data.csv**, which contains biomechanical attributes and classifications of patients as either normal or abnormal.

## Objectives
- Perform **exploratory data analysis (EDA)** to understand the dataset.
- Apply **unsupervised learning (clustering)** to explore natural groupings.
- Implement **supervised classification** to predict spinal conditions.
- Compare the insights gained from both methods.

## Methods Used
### 1. Unsupervised Learning
- **Clustering Algorithm**: Applied **K-Means clustering** to identify inherent patterns in the data.
- Evaluated clustering performance using **silhouette score and inertia**.
- Visualized cluster distributions.

### 2. Supervised Learning
- **Classification Algorithm**: Implemented a **Random Forest Classifier** for predictive modeling.
- Used **train-test split** for model evaluation.
- Performance metrics include **accuracy, precision, recall, and F1-score**.

## Project Structure
```
├── final_code.ipynb  # Compiled code with complete analysis
├── Pre-processing and initial EDA.ipynb  # Initial data exploration and preprocessing
├── Supervised.ipynb  # Supervised classification methods
├── Unsupervised Clustering.ipynb  # Clustering analysis
├── data.csv  # Dataset
├── README.md  # Project documentation
```

## How to Run
1. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Open the Jupyter Notebook files in sequence:
   - **Pre-processing and initial EDA.ipynb** for data exploration.
   - **Unsupervised Clustering.ipynb** for clustering analysis.
   - **Supervised.ipynb** for classification modeling.
   - **final_code.ipynb** for the compiled version.
3. Run the cells sequentially to perform the analysis.
4. The results, including visualizations and metrics, will be displayed.

## Results Summary
- **K-Means clustering** identified two distinct clusters, but the separation was not entirely clear.
- **Random Forest classification** achieved an **accuracy of X%** (replace with actual result).
- Feature importance analysis showed **feature_name** (replace) as the most significant predictor.

## Contributors
- #### Group Members:
  - @b-austin
  - @charlieuns
  - @charlieuns
  - @piyushsinghoffice
  - @sarahsorous

## Submission Details
- **Course**: Statistics and Machine Learning 2
- **Instructor**: Prof. Lorenzo Pellis
- **Deadline**: 18 March 2025, 3:00 PM

## License
This project is for academic purposes as part of the coursework for the **University of Manchester**.