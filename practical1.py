import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1.Load the dataset
df=pd.read_csv("netflix_titles.csv")

#Display dataset
print("ORIGINAL DATASET :")
print(df.head())

# 2. Data Cleaning
df.drop_duplicates(inplace=True)

for col in ["director", "cast", "country", "rating"]:
    df[col] = df[col].fillna("Unknown")

# 3. Data Transformation
df["date_added"] = pd.to_datetime(
    df["date_added"], errors="coerce"
)

df["date_added"] = df["date_added"].fillna(
    df["date_added"].mode()[0]
)

df["added_year"] = df["date_added"].dt.year

df["duration"] = df["duration"].fillna(
    df["duration"].mode()[0]
)

# 4. Feature Encoding
le = LabelEncoder()

for col in ["type", "country", "rating"]:
    df[col] = le.fit_transform(df[col].astype(str))

# 5. Dataset Preparation
X = df[["release_year", "added_year", "country", "rating"]]
y = df["type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Display Output
print("Cleaned Dataset:")
print(df.head())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nData Preprocessing Completed Successfully!")
