import pandas as pd
import numpy as np

# File paths (update if needed)
drug_shortage = "/Users/nareshkumardugginapalli/SQL/python/DrugShortages.csv"

# Read the files using `sep='~'`
df_shortages = pd.read_csv(drug_shortage, sep=',', dtype=str)

df_shortages= df_shortages[["Proprietary Name","Generic Name","Drug Dosage Form","Drug Strength","Therapeutic Category","Company Name","Status","Availability Information","Change Date","Date Discontinued","Date of Update","Initial Posting Date","Reason for Shortage","Related Information"]].drop_duplicates()
df_shortages.loc[df_shortages["Proprietary Name"].isna(), "Proprietary Name"] = df_shortages.loc[df_shortages["Proprietary Name"].isna(), "Generic Name"]
df_shortages.to_excel("/Users/nareshkumardugginapalli/UCC/6611/Final datasets/shortages.xlsx")
df_shortages = df_shortages.where(pd.notnull(df_shortages), "NONE")

#-----------------------------------------Superbase Login--------------------------------------------------------------------
from supabase import create_client, Client

# Replace with your actual Supabase project info
SUPABASE_URL = "https://yxbweosfodjjncdqbcnj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl4Yndlb3Nmb2Rqam5jZHFiY25qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NjY5Mjc5NiwiZXhwIjoyMDYyMjY4Nzk2fQ.yll3yzbPcWEyeT0BPOyJDoGvxuK2sKOHbLFPmTsd0C4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

#  # ----------------------------------------- save to CSV files----------------------------------------------------------------
# Convert DataFrames to list of records
shortages_records = df_shortages.to_dict(orient="records")

supabase.rpc("clear_shortages_table").execute()
print("Shortages table cleared.")

supabase.table("shortages").insert(shortages_records).execute()
print("Shortages Data uploaded to Supabase successfully.")
#
# supabase.rpc("update_therapeutic_class_shortages").execute()
# print("Updated therapeutic_class in shortages table.")