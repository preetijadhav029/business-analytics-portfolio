# import pandas as pd
# import ollama
# import json
# import time
# import os
# import subprocess
# from tqdm import tqdm
#
# # --- Configurations ---
# BASE_DIR = "/Users/nareshkumardugginapalli/SQL/python/"
# MODEL_NAME = "mistral"
# INPUT_CSV = os.path.join(BASE_DIR, "Book2.csv")
# OUTPUT_CSV = os.path.join(BASE_DIR, "classified_ingredients.csv")
# FAILED_LOG = os.path.join(BASE_DIR, "failed_ingredients.txt")
#
#
# # --- Verify Ollama is Running ---
# def check_ollama():
#     try:
#         subprocess.run(["ollama", "--version"], check=True, capture_output=True)
#         models = subprocess.run(["ollama", "list"], capture_output=True, text=True)
#         if MODEL_NAME not in models.stdout:
#             print(f"Downloading {MODEL_NAME} model...")
#             subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
#         return True
#     except Exception as e:
#         print(f"Ollama error: {str(e)}")
#         print("Install Ollama from https://ollama.com/download and run:\n  ollama pull mistral")
#         return False
#
#
# if not check_ollama():
#     exit(1)
#
# # --- Load Ingredients ---
# print(f"\nLoading ingredients from {INPUT_CSV}...")
# ingredients_df = pd.read_csv(INPUT_CSV)
# print(f" Found {len(ingredients_df)} ingredients")
#
# # --- Classification Function ---
# def classify_with_retry(ingredient, max_retries=3):
#     for attempt in range(max_retries):
#         try:
#             response = ollama.chat(
#                 model=MODEL_NAME,
#                 messages=[{
#                     'role': 'user',
#                     'content': f"""
# You are a pharmaceutical domain expert.
#
# Classify the following drug ingredient into a broad therapeutic class (e.g., Antibiotic, Antiviral, Antihypertensive, Antidiabetic, Analgesic, Antidepressant, Antifungal, etc.).
#
# Return the result in the following JSON format:
# {{
#   "therapeutic_class": "broad therapeutic class based on primary use",
#   "use": "concise primary medical use in under 10 words"
# }}
#
# Drug Ingredient: {ingredient}
# """
#                 }],
#                 format='json',
#                 options={'temperature': 0.2}
#             )
#             return json.loads(response['message']['content'])
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 print(f"Failed after {max_retries} attempts for: {ingredient}")
#                 return {"therapeutic_class": "Unknown", "use": "Unknown"}
#             time.sleep(1.5 * (2 ** attempt))  # exponential backoff
#
# # --- Process All Ingredients ---
# results = []
# failed = []
#
# print("\n🔍 Classifying ingredients...")
# for _, row in tqdm(ingredients_df.iterrows(), total=len(ingredients_df)):
#     ingredient = row["Ingredient"]
#     if pd.isna(ingredient):
#         continue
#
#     result = classify_with_retry(ingredient)
#     results.append({
#         "Ingredient": ingredient,
#         "Therapeutic_Class": result.get("therapeutic_class", "Unknown"),
#         "Use": result.get("use", "Unknown")
#     })
#
#     if result.get("therapeutic_class", "Unknown") == "Unknown":
#         failed.append(ingredient)
#
#     time.sleep(0.05)  # Slight delay to avoid overload
#
# # --- Save Results ---
# pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
# print(f"\nResults saved to: {OUTPUT_CSV}")
#
# # --- Log Failures ---
# if failed:
#     with open(FAILED_LOG, "w") as f:
#         f.writelines(f"{item}\n" for item in failed)
#     print(f"{len(failed)} ingredients failed. Logged to: {FAILED_LOG}")
# else:
#     print("All ingredients classified successfully!")

import pandas as pd
import ollama
import json
import time
import os
import subprocess
from tqdm import tqdm

# --- Configurations ---
BASE_DIR = "/Users/nareshkumardugginapalli/Desktop/"
MODEL_NAME = "mistral"
INPUT_CSV = os.path.join(BASE_DIR, "Ingredients.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "classified_ingredients-24.csv")
FAILED_LOG = os.path.join(BASE_DIR, "failed_ingredients.txt")


# --- Verify Ollama is Running ---
def check_ollama():
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
        models = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if MODEL_NAME not in models.stdout:
            print(f"Downloading {MODEL_NAME} model...")
            subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
        return True
    except Exception as e:
        print(f"Ollama error: {str(e)}")
        print("Install Ollama from https://ollama.com/download and run:\n  ollama pull mistral")
        return False


if not check_ollama():
    exit(1)

# --- Load Ingredients ---
print(f"\nLoading ingredients from {INPUT_CSV}...")
ingredients_df = pd.read_csv(INPUT_CSV)
print(f"Found {len(ingredients_df)} ingredients")

# --- Prompt Template ---
def create_prompt(ingredient):
    return f"""
You are a pharmaceutical expert and a drug classification specialist.

Given the name of a drug name, classify it into strictly one of the following broad therapeutic classes based on its primary medical use or pharmacological action:

1. Analgesics  
2. Antibiotics  
3. Antivirals  
4. Antifungals  
5. Cardiovascular Drugs  
6. Antacids
7. Anti-Inflammatories
8. Antianxiety Drugs
9. Antiarrhythmics
10. Antibacterials
11. Anticoagulants
12. Anticonvulsants
13. Antidepressants
14. Antidiarrheals
15. Antiemetics
16. Antihistamines
17. Antihypertensives
18. Antineoplastics
19. Antipsychotics
20. Antipyretics
21. Barbiturates
22. Beta-Blockers
23. Bronchodilators
24. Cold Cures
25. Corticosteroids
26. Cough Suppressants
27. Cytotoxics
28. Decongestants
29. Diuretics
30. Expectorants
31. Oral aid
32. Hypoglycemics (Oral)
33. Immunosuppressives
34. Laxatives
35. Muscle Relaxants
36. Sedatives
37. Sex Hormones (Female)
38. Sex Hormones (Male)
39. Sleeping Drugs
40. Thrombolytics
41. Tranquilizers
42. Vitamins
43. Medical Kit
44. Hormones

If the ingredient belongs to multiple classes, assign the dominant therapeutic class and append "-multi_class" at the end.

Return the result in the following JSON format:
{{
  "therapeutic_class": "one of the 15 classes (with -multi_class if applicable)",
}}

Now classify this ingredient: {ingredient}
"""

# --- Classification Function with Retry ---
def classify_with_retry(ingredient, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{
                    'role': 'user',
                    'content': create_prompt(ingredient)
                }],
                format='json',
                options={'temperature': 0.2}
            )
            return json.loads(response['message']['content'])
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts for: {ingredient}")
                return {"therapeutic_class": "Unknown"}
            time.sleep(1.5 * (2 ** attempt))  # exponential backoff

# --- Process Ingredients ---
results = []
failed = []

print("\n🔍 Classifying ingredients...")
for _, row in tqdm(ingredients_df.iterrows(), total=len(ingredients_df)):
    ingredient = row.get("Ingredient")
    if pd.isna(ingredient):
        continue

    result = classify_with_retry(ingredient)
    results.append({
        "Ingredient": ingredient,
        "Therapeutic_Class": result.get("therapeutic_class", "Unknown"),
    })

    if result.get("therapeutic_class", "Unknown") == "Unknown":
        failed.append(ingredient)

    time.sleep(0.05)  # small delay to reduce load

# --- Save Results ---
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"\nClassification complete. Results saved to: {OUTPUT_CSV}")

# --- Log Failures ---
if failed:
    with open(FAILED_LOG, "w") as f:
        f.writelines(f"{item}\n" for item in failed)
    print(f" {len(failed)}  ingredients failed. Logged to: {FAILED_LOG}")
else:
    print("All ingredients classified successfully!")
