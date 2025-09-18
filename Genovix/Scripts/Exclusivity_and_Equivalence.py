import pandas as pd
import numpy as np

# File paths (update if needed)
exclusivity_file = "/Users/nareshkumardugginapalli/SQL/python/exclusivity.txt"
products_file = "/Users/nareshkumardugginapalli/SQL/python/products.txt"
patent_file = "/Users/nareshkumardugginapalli/SQL/python/patent.txt"
drug_shortage = "/Users/nareshkumardugginapalli/SQL/python/DrugShortages.csv"

# Read the files using `sep='~'`
df_exclusivity = pd.read_csv(exclusivity_file, sep='~', dtype=str)
df_products = pd.read_csv(products_file, sep='~', dtype=str)
df_patent = pd.read_csv(patent_file, sep='~', dtype=str)
df_shortages = pd.read_csv(drug_shortage, sep=',', dtype=str)

# Merge products and exclusivity first (left join keeps all products)
merged_df = pd.merge(df_products, df_exclusivity, on="Appl_No", how="left")

# Merge the result with patents
merged_df = pd.merge(merged_df, df_patent, on="Appl_No", how="left")

merged_df.to_excel("/Users/nareshkumardugginapalli/UCC/6611/Final datasets/merged dataset.xlsx")

EE = merged_df[["Ingredient","Applicant","Exclusivity_Date","Trade_Name","Type","Exclusivity_Code","DF;Route","Strength", "TE_Code","Patent_Expire_Date_Text","Approval_Date"]].drop_duplicates()
EE = EE[EE["Type"] != "DISCN"]
EE.columns = EE.columns.str.lower()
EE["te_code"] = EE["te_code"].where(pd.notnull(EE["te_code"]), "A")
EE["ex_date"]=EE["exclusivity_date"]
# EE.loc[EE["ex_date"].isna(), "ex_date"] = EE.loc[EE["ex_date"].isna(), "patent_expire_date_text"]

EE["ex_date"] = pd.to_datetime(EE["ex_date"], errors='coerce')
EE["approval_date"] = pd.to_datetime(EE["approval_date"], errors='coerce')
#
mask = EE["ex_date"].isna() & EE["approval_date"].notna()
EE.loc[mask, "ex_date"] = EE.loc[mask, "approval_date"]

EE["exclusivity_code"] = EE["exclusivity_code"].fillna("Not Specified")

EE["days_to_exclusivity_expiry"] = (EE["ex_date"] - pd.Timestamp.today().normalize()).dt.days
EE = EE.dropna(subset=["days_to_exclusivity_expiry"])
EE["status"] = np.where(EE["days_to_exclusivity_expiry"] > 0, "yet to expire", "expired")
for col in EE.select_dtypes(include=['datetime64[ns]']).columns:
    EE[col] = EE[col].dt.strftime('%Y-%m-%d')
EE = EE.where(pd.notnull(EE), "NONE")
EE.to_excel("/Users/nareshkumardugginapalli/UCC/6611/Final datasets/EE dataset.xlsx")

#-----------------------------------------Superbase Login--------------------------------------------------------------------
from supabase import create_client, Client

# Replace with your actual Supabase project info
SUPABASE_URL = "https://yxbweosfodjjncdqbcnj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl4Yndlb3Nmb2Rqam5jZHFiY25qIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NjY5Mjc5NiwiZXhwIjoyMDYyMjY4Nzk2fQ.yll3yzbPcWEyeT0BPOyJDoGvxuK2sKOHbLFPmTsd0C4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

 # ----------------------------------------- save to CSV files----------------------------------------------------------------
# Convert DataFrames to list of records
EE_records = EE.to_dict(orient="records")

supabase.rpc("clear_ee_table").execute()
print("EE table cleared.")

supabase.table("ee").insert(EE_records).execute()
print("All data uploaded to Supabase successfully.")

supabase.rpc("update_therapeutic_class_ee").execute()
print("Updated therapeutic_class in ee")