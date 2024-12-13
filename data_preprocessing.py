from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Progress bar for preprocessing steps
    with tqdm(total=4, desc="Preprocessing Data") as pbar:
        # Strip extra spaces from column names
        df.columns = df.columns.str.strip()
        pbar.update(1)
        
        if 'Air Quality' in df.columns:
            target_mapping = {
                'Good': 0,
                'Moderate': 1,
                'Poor': 2,
                'Hazardous': 3
            }
            df['Air_Quality_Levels'] = df['Air Quality'].map(target_mapping)
            pbar.update(1)
        else:
            raise KeyError("Column 'Air Quality' not found in the dataset.")

        # Drop the original target column
        df = df.drop(columns=['Air Quality'])
        pbar.update(1)
        
        # Train-test split
        X = df.drop(columns=['Air_Quality_Levels'])
        y = df['Air_Quality_Levels']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pbar.update(1)
    
    print("Preprocessing completed!")
    return X_train, X_test, y_train, y_test, target_mapping