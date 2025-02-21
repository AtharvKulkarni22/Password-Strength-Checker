import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_dir, "..", "Dataset", "data.csv")
cleaned_data_path = os.path.join(base_dir, "..", "Dataset", "cleaned_data.csv")

cleaned_rows = []
with open(data_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(",")
        
        if len(parts) == 2:
            password, strength = parts
            
            if strength.strip().isdigit():
                cleaned_rows.append([password, int(strength.strip())])

df = pd.DataFrame(cleaned_rows, columns=["password", "strength"])

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df.to_csv(cleaned_data_path, index=False)

print("Cleaned dataset preview:")
print(df.head(10))
