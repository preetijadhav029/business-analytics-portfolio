from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("/Users/nareshkumardugginapalli/UCC/6611/Final datasets/ML model.csv")

# Step 1: Assign filed_rating based on average approval time per year
# def assign_filed_rating(df):
#     year_avg = df.groupby('Year')['Days'].mean().to_dict()
#     def get_rating(row):
#         avg = year_avg[row['Year']]
#         lower_bound = avg - 15
#         upper_bound = avg + 15
#         if row['Days'] < lower_bound:
#             return 'high'
#         elif row['Days'] > upper_bound:
#             return 'low'
#         else:
#             return 'medium'
#     df['filed_rating'] = df.apply(get_rating, axis=1)
#     return df
#
# df = assign_filed_rating(df)
group_cols = [ "Type", "SubmissionType", "Sub Class"]

# Compute group-wise average Days
group_avg = df.groupby(group_cols)['Days'].transform('mean')

# Define the rating logic
def get_rating(row, avg):
    if row['Days'] < avg - 60:
        return 'high'
    elif row['Days'] > avg:
        return 'low'
    else:
        return 'medium'

# Apply rating logic
df['filed_rating'] = df.apply(lambda row: get_rating(row, group_avg[row.name]), axis=1)

# Step 2: Filter out rare target classes
class_counts = df['filed_rating'].value_counts()
valid_classes = class_counts[class_counts > 1].index
df = df[df['filed_rating'].isin(valid_classes)]

# Step 3: Define features and target (removed 'Applicant')
feature_cols = ["Therapeutic class", "Type", "SubmissionType", "Sub Class", "Year"]
X = df[feature_cols]
y = df["filed_rating"]

# Step 4: Encode categorical variables
encoders = {}
X_encoded = pd.DataFrame()
for col in X.columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    encoders[col] = le

# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, stratify=y, test_size=0.3, random_state=42
)

# Step 6: Random Forest with tuned hyperparameters
model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4
)
model.fit(X_train, y_train)

# Step 7: Evaluate on test set
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 8: Cross-validation with StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_encoded, y, cv=skf, scoring='accuracy')
print(f"\nStratified CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Std CV Accuracy: {np.std(cv_scores):.4f}")

# Step 10: Predict rating for 2025
years_to_predict = [2025]
base_combinations = df[["Therapeutic class", "Type", "SubmissionType", "Sub Class"]].drop_duplicates()
future_rows = []
for year in years_to_predict:
    temp = base_combinations.copy()
    temp["Year"] = year
    future_rows.append(temp)

future_df = pd.concat(future_rows, ignore_index=True)

# Encode future data
future_encoded = pd.DataFrame()
for col in feature_cols:
    le = encoders[col]
    unseen_labels = set(future_df[col]) - set(le.classes_)
    if unseen_labels:
        print(f"Adding unseen labels to encoder for column '{col}': {unseen_labels}")
        le.classes_ = np.append(le.classes_, list(unseen_labels))
    future_encoded[col] = le.transform(future_df[col])

# Predict future ratings
future_df["predicted_rating"] = model.predict(future_encoded)

# Step 11: Save predictions
future_df.to_csv("/Users/nareshkumardugginapalli/UCC/6611/Final datasets/predicted DB upload.csv", index=False)
print("Predictions saved to 'predicted_filed_ratings_2025_updated.xlsx'")
