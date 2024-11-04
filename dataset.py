from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# Load dataset
dataset = pd.read_csv('DataSet.csv')


# Select features and target
X = dataset.iloc[:, :-1].values
XX = dataset.iloc[:, :-1].values
XXX = dataset.iloc[:, :-1].values
# Adjust if necessary based on your target column
Y = dataset.iloc[:, 3].values

# Missing Value Handeling by MEAN
# Create an imputer for the specific columns
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Fit and transform the imputer on the specified columns (1 and 2 in this case)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])


# Missing Value Handeling by MEDIAN
# Create an imputer for the specific columns
imputer = SimpleImputer(missing_values=np.nan, strategy="median")

# Fit and transform the imputer on the specified columns (1 and 2 in this case)
XX[:, 1:3] = imputer.fit_transform(XX[:, 1:3])


# Missing Value Handeling by most_frequent
# Create an imputer for the specific columns
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

# Fit and transform the imputer on the specified columns (1 and 2 in this case)
XXX[:, 1:3] = imputer.fit_transform(XXX[:, 1:3])


# Concept of Dummy Varialble, Handeling the conflict of them
# Encoding categorical features
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# One Hot Encoding for the categorical feature
column_transformer = ColumnTransformer(
    transformers=[
        # One-hot encode the first column (index 0)
        ('onehot', OneHotEncoder(), [0])
    ],
    remainder='passthrough'  # Keep the other columns unchanged
)

X = column_transformer.fit_transform(X)

# Now X contains the one-hot encoded categorical variable and imputed values


# Training and Testing Data (divide the data into two parts)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)


# Standard and fit the data for better prediction
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)
