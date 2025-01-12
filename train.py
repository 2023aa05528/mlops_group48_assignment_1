# Import necessary libraries
import joblib
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the Penguin dataset
print("Loading the dataset")
penguins = sns.load_dataset("penguins")

# Preprocessing
# Drop rows with missing values
print("Pre-processing the dataset")
penguins = penguins.dropna()

# Encode species as labels
label_encoder = LabelEncoder()
penguins['species'] = label_encoder.fit_transform(penguins['species'])

# One-hot encode categorical variables (island and sex)
penguins = pd.get_dummies(penguins, columns=['island', 'sex'], drop_first=True)

# Define features (X) and target (y)
X = penguins.drop('species', axis=1)
y = penguins['species']

# Train-test split
print("Splitting the dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
print("Training the model")
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

print('Saving the model')
joblib.dump(rf_classifier,'models/model.pkl')