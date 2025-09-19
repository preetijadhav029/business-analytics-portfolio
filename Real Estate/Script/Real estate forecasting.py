import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 #open file
df = pd.read_csv("C:/Users/Preeti/OneDrive/Desktop/Projects/Real Estate Sales Forecasting – Dublin/Ireland House Price Final.csv")
 #data cleaning
 #renaming columns
df.rename(columns={"buying or not buying": "buying_decision",
                   "Renovation needed": "renovation",
                   "price-per-sqft-$": "price_per_sqft"}, inplace=True)
# Filling missing values in 'balcony' column with 0
df['balcony'] = df['balcony'].fillna(0)
 # Converting 'buying_decision' values to binary (Yes=1, No=0)
df['buying_decision'] = df['buying_decision'].map({'Yes': 1, 'No': 0})
 # Mapping 'renovation' values to numerical scores
df['renovation']=df['renovation'].map({'Yes':1, 'Maybe':0.5,'No':0})
 # ensuring price_per_sqft is a float
df['price_per_sqft'] = df['price_per_sqft'].astype(float).round(2)
 # dropping rows with missing values from size and location
df.dropna(subset=['location'], inplace=True)
df.dropna(subset=['size'], inplace=True)
 #print("Before Cleaning:")
 #print( df.isnull().sum())
df2 = df.dropna()
 #print("After Cleaning:")
 #print( df2.isnull().sum())
df.groupby('property_scope')['property_scope'].describe()
 #print(df.groupby('property_scope')['property_scope'].value_counts())
 # Extracting numeric values from size to integers
df['size'] = df['size'].str.extract(r'(\d+)').astype(int)
 #print(df2.total_sqft.unique())
df2['total_sqft'].astype(float)
def convert_sqft_to_num(x):
    tokens= x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df3=df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert_sqft_to_num)
 #print(df3.head())
 #rename locations
df3.location = df3.location.apply(lambda x: x.strip())
 # Creating a mapping for unique locations
df3['location'] = df3['location'].astype(str)  # Ensure all values are strings
location_mapping = {location: idx + 1 for idx, location in
enumerate(df3['location'].unique())}
 # Replacing location names with their corresponding numbers
df3['location'] = df3['location'].map(location_mapping)
print("Location Mapping:")
print(location_mapping)
 # Creating a mapping for unique BER
df3['BER'] = df3['BER'].astype(str)  # Ensure all values are strings
BER_mapping = {BER: idx + 1 for idx, BER in enumerate(df3['BER'].unique())}
 # Replacing location names with their corresponding numbers
df3['BER'] = df3['BER'].map(BER_mapping)
print("BER Mapping:")
print(BER_mapping)
 #removing columns
df3 = df3.drop(["ID", "property_scope", "availability"], axis='columns')
 #outliers removal
 #checking    outliers
df4 = df3[(df3.total_sqft/df3.size<300)]
print()
print('Total rows and columns before removing outliers:' , df4.shape)
 #using IQR
Q1 = df4['price_per_sqft'].quantile(0.25)  # 25th percentile
Q3 = df4['price_per_sqft'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1  # Interquartile Range
 # Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
 # Filter out outliers
df4 = df4[(df4['price_per_sqft'] >= lower_bound) & (df4['price_per_sqft'] <=
upper_bound)]
 #print(df4.price_per_sqft.describe())
 #removing price_per_sqft per location using standard deviation
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.total_sqft)
        st = np.std(subdf.total_sqft)
        reduced_df = subdf[(subdf.total_sqft > (m-st)) & (subdf.total_sqft <=
(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
df5 = remove_pps_outliers(df4)
print('Total rows and columns after removing outliers:', df5.shape)
# clean size column
df5['size'] = df5['size'].str.extract(r'(\d+)').astype(float)
 # Scatter plot function
def plot_scatter_chart(df5, location):
    # Filter data for each BHK type
    bhk1 = df5[(df5['location'] == location) & (df5['size'] == 1)]
    bhk2 = df5[(df5['location'] == location) & (df5['size'] == 2)]
    bhk3 = df5[(df5['location'] == location) & (df5['size'] == 3)]
    bhk4 = df5[(df5['location'] == location) & (df5['size'] == 4)]
    bhk5 = df5[(df5['location'] == location) & (df5['size'] == 5)]
    plt.rcParams['figure.figsize'] = (15, 10)
    # Plot data if available
    if not bhk1.empty:
        plt.scatter(bhk1['total_sqft'], bhk1['price_per_sqft'], color='green',
label='1 BHK', s=50)
    if not bhk2.empty:
        plt.scatter(bhk2['total_sqft'], bhk2['price_per_sqft'], color='blue',
label='2 BHK', s=50)
    if not bhk3.empty:
        plt.scatter(bhk3['total_sqft'], bhk3['price_per_sqft'], color='red',
marker='v', label='3 BHK', s=50)
    if not bhk4.empty:
        plt.scatter(bhk4['total_sqft'], bhk4['price_per_sqft'], color='yellow',
marker='p', label='4 BHK', s=50)
    if not bhk5.empty:
        plt.scatter(bhk5['total_sqft'], bhk5['price_per_sqft'], color='purple',
marker='x', label='5 BHK', s=50)
    plt.xlabel('Total sqft')
    plt.ylabel('Price per sqft')
    plt.title(f"Property Prices in Location {location}")
    plt.legend()
    plt.show()
 # Example usage
plot_scatter_chart(df5, location=2)
 # Feature engineering
df5['total_price'] = df5['price_per_sqft'] * df5['total_sqft'].astype(float)
df5['total_price'] = df5['total_price'].round(2)
correlation_matrix = df5.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 # Assuming df5 is your DataFrame and the relevant columns are present
# Define the dependent variable (target) and independent variables (features)
X = df5[['bath', 'balcony', 'BER', 'renovation', 'size']]  # Independent variables
y = df5['total_price']  # Dependent variable
 # Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
 # Create the linear regression model
model = LinearRegression()
print('-------------Linear regression---------------')
 # Train the model with the training data
model.fit(X_train, y_train)
 # Make predictions on the test set
y_pred = model.predict(X_test)
 # Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared
 # Print the results
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
 # Optionally, print the coefficients and intercept of the model
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
from sklearn.model_selection import cross_val_score
 # Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)  # Negating MSE to match convention
print(f"Cross-validation scores: {[f'{score * 100:.2f}%' for score in cv_scores]}")
print(f"Average CV score:{cv_scores.mean()*100:.2f}%")
print('-------------logistic regression---------------')
 # logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
 # Assuming df5 is your DataFrame and the relevant columns are present
 # Define the dependent variable (target) and independent variables (features)
X = df5[['bath', 'balcony', 'BER', 'renovation', 'size', 'total_sqft']]
#Independent variables
y = df5['buying_decision']  # Dependent variable (binary)
 # Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
 # Create the logistic regression model
model = LogisticRegression(max_iter=250)
 # Train the model with the training data
model.fit(X_train, y_train)
 # Make predictions on the test set
y_pred = model.predict(X_test)
 # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Accuracy of the model
conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix
class_report = classification_report(y_test, y_pred)  # Classification report
 # Print the results
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
print('-------------SVM---------------')
 #SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
 # Assuming df5 is your DataFrame and the relevant columns are present
 # Define the independent variables (features) and the dependent variable (target)
X = df5[['bath', 'balcony', 'BER', 'renovation', 'size', 'total_sqft']]
#Independent variables
y = df5['buying_decision']  # Dependent variable (target)
 # Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
 # Standardize the features (important for SVM models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 # Create the SVM classifier model
model = SVC(kernel='linear', class_weight='balanced')
 # You can experiment withdifferent kernels like 'rbf' or 'poly'
 # Train the model
model.fit(X_train_scaled, y_train)
 # Make predictions
y_pred = model.predict(X_test_scaled)
 # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=1))
 #decision tree
print('----------------decision tree------------')
from sklearn.tree import DecisionTreeClassifier
 # Assuming df5 is your DataFrame and the relevant columns are present
 # Define the independent variables (features) and dependent variable (target)
X = df5[['bath', 'balcony', 'BER', 'renovation', 'size','location']]
# Independent variables
y = df5['buying_decision']  # Dependent variable (target)
 # Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 # Create the decision tree model
model = DecisionTreeClassifier()
 # Train the model
model.fit(X_train, y_train)
 # Make predictions on the test set
y_pred = model.predict(X_test)
 # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Accuracy
print(f'Accuracy: {accuracy}')
 # Print classification report and confusion matrix for more detailed evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
 #random forest
print('---------------random forest----------------')
from sklearn.ensemble import RandomForestClassifier
 # Assuming df5 is your DataFrame and the relevant columns are present
 # Define the independent variables (features) and dependent variable (target)
X = df5[['bath', 'balcony', 'BER', 'renovation', 'size','location']]
 # Independent variables
y = df5['buying_decision']  # Dependent variable (target)
 # Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
 # Create the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100)
 # Train the model
rf_model.fit(X_train, y_train)
 # Make predictions on the test set
y_pred = rf_model.predict(X_test)
 # Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
# Print detailed classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))
 # Print confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
 #naivebayes
print('-----------------Naive bayes---------------------')
from sklearn.naive_bayes import GaussianNB
 # Assuming df5 is your DataFrame and the relevant columns are present
 # Define the independent variables (features) and dependent variable (target)
X = df5[['bath', 'balcony', 'BER', 'renovation', 'size', 'total_sqft']]
#Independent variables
y = df5['buying_decision']  # Dependent variable
 # Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
 # Create the Naive Bayes model (Gaussian Naive Bayes for continuous features)
model = GaussianNB()
 # Train the model with the training data
model.fit(X_train, y_train)
 # Make predictions on the test set
y_pred = model.predict(X_test)
 # Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)  # Accuracy
report = classification_report(y_test, y_pred)  # Detailed classification report
 # Print the results
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
 #Ensemble with voting classifier
print('----------------Voting Classifier--------------------')
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 # Data Preparation
x = df5.drop(['buying_decision', 'price_per_sqft'], axis='columns')
y = df5['buying_decision']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=42)
 # Define individual base models
bagging_dt = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10,
random_state=0)
bagging_lr = BaggingClassifier(estimator=LogisticRegression(max_iter=500),
n_estimators=10, random_state=0)
bagging_rf = BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=10,
random_state=0)
 # Combine models using VotingClassifier
voting_model = VotingClassifier(
    estimators=[
        ('Bagging_DT', bagging_dt),
        ('Bagging_LR', bagging_lr),
        ('Bagging_RF', bagging_rf),
    ],
    voting='hard'
)
 # Train Voting Classifier
voting_model.fit(x_train, y_train)
 # Evaluate the combined model
y_pred = voting_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Combined Voting Model Accuracy: {accuracy:.2f}")
 # Model accuracy comparison
models = ['Logistic Regression', 'SVM', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Voting Classifier']
accuracies = [65.61, 67, 67.03, 74.06, 76.26, 73]  # Replace with actual accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracies)
plt.title('Model Accuracy Comparison', fontsize=16)
plt.ylabel('Accuracy (%)')
plt.xlabel('Models')
plt.xticks(rotation=45)
plt.show()
 # Feature Importance from Random Forest
importances = rf_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance (Random Forest)', fontsize=16)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()