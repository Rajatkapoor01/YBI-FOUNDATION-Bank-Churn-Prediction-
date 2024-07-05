# YBI-FOUNDATION-Bank-Churn-Prediction-
The objective of this project is to cluster bank customers and predict customer churn using machine learning techniques.

### **Project Summary: Customer Clustering and Churn Prediction in a Bank**

#### **Objective:**
The main goal of this project is to segment bank customers and predict customer churn using machine learning techniques. By identifying patterns and predicting churn, banks can devise strategies to retain customers and improve their services.

#### **Data Source:**
The dataset used for this project is available on Kaggle: [Bank Customer Churn Modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling).

#### **Libraries and Tools:**
- **Data Manipulation and Analysis:** NumPy, Pandas, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, TensorFlow, Keras
- **Miscellaneous:** Random, mpl_toolkits.mplot3d, Matplotlib patches

#### **Data Preprocessing:**
1. **Loading the Dataset:** 
   - The dataset is loaded from a CSV file.
2. **Dropping Unnecessary Columns:** 
   - Columns like `RowNumber`, `CustomerId`, and `Surname` are dropped.
3. **Renaming and Encoding:**
   - The `Gender` column is renamed to `IsMale` and encoded to binary (1 for Male, 0 for Female).
4. **Separating Numerical and Categorical Variables:**
   - Numerical variables and categorical variables are separated for further processing.
5. **One-Hot Encoding:**
   - Categorical variables are one-hot encoded.
6. **Normalization:**
   - Numerical variables are normalized to a range between 0 and 1.

#### **Data Visualization:**
- **Distribution of Variables:** Histograms are used to visualize the distribution of various numerical and categorical variables.
- **Nationality Proportion:** A pie chart is used to show the proportion of customers from Germany, Spain, and France.
- **Gender Ratio:** A pie chart displays the gender distribution among customers.
- **Correlation Matrix:** A heatmap is used to visualize the correlation between different variables in the dataset.

#### **Model Training:**
1. **Defining Features and Target Variable:**
   - The features (`X`) and the target variable (`Y`) are defined from the processed dataset.
2. **Train-Test Split:**
   - The dataset is split into training and testing sets.
3. **Model Architecture:**
   - A neural network model is built using TensorFlow and Keras with multiple layers (Dense, BatchNormalization, Dropout).
4. **Model Compilation and Training:**
   - The model is compiled with the Adam optimizer and trained using binary cross-entropy loss.
5. **Model Evaluation:**
   - The model's performance is evaluated using accuracy, and a confusion matrix is plotted for further insights.

#### **Key Insights:**
1. **Nationality Distribution:** 
   - The majority of customers are from France, followed by Germany and Spain.
2. **Gender Distribution:** 
   - The dataset contains more male customers compared to female customers.
3. **Correlation Analysis:** 
   - The heatmap reveals correlations between different features, helping to understand which features are strongly related.

#### **Model Performance:**
- **Accuracy:** The model's accuracy is evaluated after each iteration of training, with results indicating the model's capability to predict churn accurately.
- **Confusion Matrix:** The confusion matrix helps in understanding the model's performance in terms of true positives, true negatives, false positives, and false negatives.

#### **Conclusion:**
This project demonstrates how to preprocess a dataset, visualize key insights, and build a machine learning model to predict customer churn. By understanding customer segments and predicting churn, banks can implement targeted strategies to retain customers and enhance their overall experience. Further model evaluation metrics like precision, recall, F1 score, and ROC-AUC curves can be used to gain deeper insights into the model's performance.

### **SCREEN SHOT**
![SCREENSHOT1](https://github.com/Rajatkapoor01/YBI-FOUNDATION-Bank-Churn-Prediction-/blob/main/ScreenShot/bankPred1.png)
![SCREENSHOT2](https://github.com/Rajatkapoor01/YBI-FOUNDATION-Bank-Churn-Prediction-/blob/main/ScreenShot/bankPred2.png)
![SCREENSHOT3](https://github.com/Rajatkapoor01/YBI-FOUNDATION-Bank-Churn-Prediction-/blob/main/ScreenShot/bankPred3.png)
![SCREENSHOT4](https://github.com/Rajatkapoor01/YBI-FOUNDATION-Bank-Churn-Prediction-/blob/main/ScreenShot/bankPred4.png)
