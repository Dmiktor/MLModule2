import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Debug: print the column names
    print("Column names in the dataset:", df.columns)

    # Clean the 'Air Quality' column
    df['Air Quality'] = df['Air Quality'].str.strip().str.lower()

    # Define the target mapping
    target_mapping = {
        'good': 0,
        'moderate': 1,
        'poor': 2,
        'hazardous': 3
    }

    # Debug: check unique values before mapping
    print("Unique values in 'Air Quality' before mapping:", df['Air Quality'].unique())

    # Apply the mapping to the target column
    df['Air_Quality_Levels'] = df['Air Quality'].map(target_mapping)

    # Debug: check for NaNs after mapping
    print("Unique values in 'Air_Quality_Levels' after mapping:", df['Air_Quality_Levels'].unique())
    print("Rows with NaN in 'Air_Quality_Levels':")
    print(df[df['Air_Quality_Levels'].isna()])

    # Drop rows with NaN in target
    df = df.dropna(subset=['Air_Quality_Levels'])

    # Separate features and target
    X = df.drop(columns=['Air Quality', 'Air_Quality_Levels'])
    y = df['Air_Quality_Levels']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, target_mapping