import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("predictions.db")

# Read all records from the predictions table
df = pd.read_sql_query("SELECT * FROM predictions", conn)

conn.close()

# Print the DataFrame
print(df)
