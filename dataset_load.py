import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# =====================================================
# Step 1 — Load + Tag Dataset
# =====================================================

def load_cmapss(file_path, dataset_name):
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    columns = ['unit', 'cycle',
               'op1', 'op2', 'op3'] + \
              [f'sensor{i}' for i in range(1, 22)]

    df.columns = columns
    df['dataset'] = dataset_name   # 🔴 important

    return df

df1 = load_cmapss("train_FD001.txt", "FD001")
df2 = load_cmapss("train_FD002.txt", "FD002")
df3 = load_cmapss("train_FD003.txt", "FD003")
df4 = load_cmapss("train_FD004.txt", "FD004")

# =====================================================
# Step 1 — Merge all datasets
# =====================================================

df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# =====================================================
# 🔴 Create dataset-aware unit_id (teacher requirement)
# =====================================================

df['unit_id'] = df['dataset'] + "_" + df['unit'].astype(str)

print("Total engines:", df['unit_id'].nunique())

# =====================================================
# Step 3 — Remove useless columns (FIRST)
# =====================================================

constant_cols = [
    'sensor1','sensor5','sensor6','sensor10',
    'sensor16','sensor18','sensor19','op3'
]

df = df.drop(columns=constant_cols)

# =====================================================
# Step 4 — Compute RUL (using unit_id)
# =====================================================

max_cycle = df.groupby('unit_id')['cycle'].max().reset_index()
max_cycle.columns = ['unit_id', 'max_cycle']

df = df.merge(max_cycle, on='unit_id')
df['RUL'] = df['max_cycle'] - df['cycle']
df.drop(columns=['max_cycle'], inplace=True)

# =====================================================
# Step 5 — Engine-wise split
# =====================================================

engine_ids = df['unit_id'].unique()

train_engines, val_engines = train_test_split(
    engine_ids,
    test_size=0.2,
    random_state=42
)

train_df = df[df['unit_id'].isin(train_engines)].copy()
val_df   = df[df['unit_id'].isin(val_engines)].copy()

print("Train engines:", len(train_engines))
print("Validation engines:", len(val_engines))

# =====================================================
# Step 6 — Normalize (fit only on train)
# =====================================================

feature_cols = ['cycle', 'op1', 'op2'] + \
               [col for col in df.columns if 'sensor' in col]

scaler = StandardScaler()

train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])

# =====================================================
# Step 7 — Rolling Window (per engine)
# =====================================================

def create_sequences(dataframe, seq_length, feature_cols):
    X, y = [], []

    for unit in dataframe['unit_id'].unique():
        unit_df = dataframe[dataframe['unit_id'] == unit]

        data = unit_df[feature_cols].values
        rul = unit_df['RUL'].values

        for i in range(len(unit_df) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(rul[i+seq_length])

    return np.array(X), np.array(y)

SEQ_LENGTH = 20

X_train, y_train = create_sequences(train_df, SEQ_LENGTH, feature_cols)
X_val, y_val     = create_sequences(val_df, SEQ_LENGTH, feature_cols)

# =====================================================
# Flatten for RandomForest
# =====================================================

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)

# =====================================================
# Train Model
# =====================================================

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_flat, y_train)

# =====================================================
# Evaluate
# =====================================================

y_pred = model.predict(X_val_flat)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print("Validation RMSE:", rmse)

# =====================================================
# Save processed dataset
# =====================================================

df.to_csv("CMAPSS_combined_processed.csv", index=False)