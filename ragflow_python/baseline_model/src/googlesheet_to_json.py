import pandas as pd

# Path to your Excel file
excel_file = "ragflow_python/baseline_model/data/raw/QA_pairs_v1.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file)

# Convert the DataFrame to JSON
# orient='records' makes each row a separate JSON object in a list
json_data = df.to_json(orient='records', indent=2)

# Save the JSON data to a file
with open("ragflow_python/baseline_model/data/processed/QA_pairs_v1.json", "w") as f:
    f.write(json_data)

print("Conversion complete! JSON saved to QA_pairs_v1.json")

