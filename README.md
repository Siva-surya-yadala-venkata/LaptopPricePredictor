# **Laptop Price Prediction Using Machine Learning and Deep Learning**

This project predicts laptop prices using a combination of Machine Learning and Deep Learning models, including **Artificial Neural Networks (ANN)**, **K-Nearest Neighbors (KNN)**, **Random Forest**, and **Decision Trees**. The focus is on building a robust prediction system and comparing the performance of various models.

---

## **Project Highlights**

1. **Data Preprocessing**:
   - Engineered features such as screen resolution, RAM interaction, and resolution score for enhanced predictions.
   - Applied one-hot encoding for categorical variables like `Company`, `TypeName`, and `OpSys`.
   - Normalized numerical features to improve model efficiency.

2. **Implemented Models**:
   - **Artificial Neural Network (ANN)**: A deep learning model with multiple hidden layers and advanced hyperparameter tuning for robust predictions.
   - **K-Nearest Neighbors (KNN)**: A simple yet effective model for finding price patterns based on similar data points.
   - **Random Forest**: A tree-based ensemble model for accurate price prediction and feature importance analysis.
   - **Decision Trees**: A single-tree model offering interpretability and quick predictions.

3. **Visualizations**:
   - Scatter plots for Actual vs. Predicted prices.
   - Correlation heatmaps for feature analysis.
   - Error distribution histograms.
   - Feature importance visualizations for Random Forest.
---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Flask
- **Deep Learning Framework**: Custom implementation of ANN
- **Tools**: Jupyter Notebook, Flask Web Framework

---

## **How to Use**
1. Clone this repository:  
   ```bash
   git clone https://github.com/yourusername/laptop-price-prediction.git
   ```
2. Navigate to the project directory and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3.Run Python main.py For Ann Model so like that we need to Use

---

## **Results**
- Compared the performance of ANN, KNN, Random Forest, and Decision Tree models.
- **ANN** achieved the most robust results, leveraging deep learning for complex feature interactions.
- Feature importance and error metrics provide insights into model behavior.
