MSRP Price Prediction

This project aims to predict the MSRP price of vehicles using their technical and categorical characteristics. Several preprocessing strategies and machine learning models were implemented and compared.

Dataset

The dataset contains vehicle information such as Engine HP, Number of Cylinders, Fuel Type, Transmission Type, Vehicle Size, Vehicle Style, Make, and Model. The target variable is MSRP.

Data Cleaning

The following steps were applied for all versions:

-Removal of extreme outliers (top 1 percent)
-Removal of the column Market Category
-Handling missing values (mean for numerical features, most frequent value for categorical features)
-Removal of highly correlated variables to reduce multicollinearity

Model Versions

V1: Baseline linear regression with basic preprocessing. Used as an initial benchmark.

V2A: Improved preprocessing and linear regression.

V2B: Same preprocessing as V2A but using log(MSRP) as the target. This reduced the MAE but lowered R².

V3: Random Forest model. This non-linear model improved performance compared to linear models. MAE around 2900 and R² around 0.977.

V4: Full comparison of multiple preprocessings (one-hot encoding, scaling, target encoding) and models (linear regression, random forest, gradient boosting, XGBoost, SVR). The best combination was Target Encoding with XGBoost, with an MAE around 2860. Random Forest remained competitive with a slightly better R² but a higher MAE.

Conclusion

Tree-based models perform better than linear models for this type of regression. Target encoding is effective for categorical variables with many unique values. The best performing model was XGBoost combined with target encoding. Random Forest also gave strong results.

Files in the repository

Project_Code_V1.ipynb
Project_Code_V2A.ipynb
Project_Code_V2B.ipynb
Project_Code_V3.ipynb
Project_Code_V4.ipynb
data.csv
requirements.txt
README.md

Running instructions

Install dependencies using:
pip install -r requirements.txt

Run the notebooks using:
jupyter notebook
