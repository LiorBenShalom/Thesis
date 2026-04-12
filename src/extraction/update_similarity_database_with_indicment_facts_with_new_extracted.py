import pandas as pd
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
import re
from openai import OpenAI
import gc
from tqdm import tqdm
import csv
import time


DRY_RUN = False  # Set to True for dry run (no updates, only comparison), False to actually update
PROCESS_ONLY_TARGET_VERDICTS = False  # Set to True to process only verdicts from target.csv
domain="wep"


# ========== API Setup ==========
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")  
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========== File Paths ==========

if domain=="drugs":
    base_path="/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/drugs/"
else:
    base_path="/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/"
csv_directory =base_path+'verdict_csv'
out_dir = base_path+"gpt"


def count_word_difference(text1: str, text2: str) -> int:
    """
    Calculate the absolute difference in word count between two texts.
    """
    if pd.isna(text1) or text1 == "":
        text1 = ""
    if pd.isna(text2) or text2 == "":
        text2 = ""
    
    words1 = set(str(text1).split())
    words2 = set(str(text2).split())
    
    # Calculate symmetric difference (words in one but not the other)
    diff_words = words1.symmetric_difference(words2)
    return len(diff_words)


def update_similarity_database(
    similarity_db_path: str,
    new_facts_csv_path: str,
    output_path: str,
    # --- Column Configuration (Change these if your CSV headers are different) ---
    sim_verdict1_col: str = 'verdict_1',        # ID column in Similarity DB
    sim_verdict2_col: str = 'verdict_2',        # ID column in Similarity DB
    sim_facts1_col: str = 'indicment_facts_1',  # Target text col in Similarity DB
    sim_facts2_col: str = 'indicment_facts_2',  # Target text col in Similarity DB
    
    new_id_col: str = 'verdict',             # ID column in New Facts CSV
    new_text_col: str = 'extracted_gpt_facts',      # Text column in New Facts CSV
    dry_run: bool = False
):
    """
    Updates the indictment facts in the similarity database using a master facts CSV.
    If dry_run is True, only generates a comparison CSV without updating the database.
    """
    print(f"📂 Loading Similarity DB: {similarity_db_path}")
    if not os.path.exists(similarity_db_path):
        print(f"❌ Error: File not found - {similarity_db_path}")
        return
    df_sim = pd.read_csv(similarity_db_path)
    
    print(f"📂 Loading New Facts CSV: {new_facts_csv_path}")
    if not os.path.exists(new_facts_csv_path):
        print(f"❌ Error: File not found - {new_facts_csv_path}")
        return
    df_facts = pd.read_csv(new_facts_csv_path)
    
    # 1. Create Lookup Dictionary (ID -> Fact)
    # Ensure IDs are strings and stripped of whitespace for accurate matching
    print("⚙️  Creating lookup map...")
    df_facts[new_id_col] = df_facts[new_id_col].astype(str).str.strip()
    
    # Remove duplicates in master file (keep last or first valid fact)
    df_facts = df_facts.drop_duplicates(subset=[new_id_col], keep='last')
    
    # Convert to dictionary
    facts_map = pd.Series(
        df_facts[new_text_col].values, 
        index=df_facts[new_id_col]
    ).to_dict()
    
    print(f"   Mapped {len(facts_map)} unique verdicts from master file.")

    # 2. Track changes for comparison
    print("⚙️  Analyzing changes...")
    
    # Convert similarity IDs to string for matching
    df_sim[sim_verdict1_col] = df_sim[sim_verdict1_col].astype(str).str.strip()
    df_sim[sim_verdict2_col] = df_sim[sim_verdict2_col].astype(str).str.strip()
    
    # Collect all changes
    changes_list = []
    
    # Check facts_1
    if sim_facts1_col in df_sim.columns:
        original_1 = df_sim[sim_facts1_col].fillna("")
        new_values_1 = df_sim[sim_verdict1_col].map(facts_map)
        new_values_1 = new_values_1.fillna("")
        
        for idx, row in df_sim.iterrows():
            verdict_id = row[sim_verdict1_col]
            old_fact = str(original_1.iloc[idx]) if not pd.isna(original_1.iloc[idx]) else ""
            new_fact = str(new_values_1.iloc[idx]) if not pd.isna(new_values_1.iloc[idx]) else ""
            
            if old_fact != new_fact and verdict_id in facts_map:
                word_diff = count_word_difference(old_fact, new_fact)
                if word_diff > 2:  # Only include changes with more than 2 words difference
                    changes_list.append({
                        'verdict_id': verdict_id,
                        'column': sim_facts1_col,
                        'old_facts': old_fact,
                        'new_facts': new_fact,
                        'word_difference': word_diff
                    })
    
    # Check facts_2
    if sim_facts2_col in df_sim.columns:
        original_2 = df_sim[sim_facts2_col].fillna("")
        new_values_2 = df_sim[sim_verdict2_col].map(facts_map)
        new_values_2 = new_values_2.fillna("")
        
        for idx, row in df_sim.iterrows():
            verdict_id = row[sim_verdict2_col]
            old_fact = str(original_2.iloc[idx]) if not pd.isna(original_2.iloc[idx]) else ""
            new_fact = str(new_values_2.iloc[idx]) if not pd.isna(new_values_2.iloc[idx]) else ""
            
            if old_fact != new_fact and verdict_id in facts_map:
                word_diff = count_word_difference(old_fact, new_fact)
                if word_diff > 2:  # Only include changes with more than 2 words difference
                    changes_list.append({
                        'verdict_id': verdict_id,
                        'column': sim_facts2_col,
                        'old_facts': old_fact,
                        'new_facts': new_fact,
                        'word_difference': word_diff
                    })
    
    # 3. Generate comparison CSV
    if changes_list:
        df_changes = pd.DataFrame(changes_list)
        comparison_output = output_path.replace('.csv', '_comparison_dry_run.csv')
        print(f"📊 Found {len(df_changes)} facts with changes > 2 words")
        print(f"💾 Saving comparison CSV to: {comparison_output}")
        df_changes.to_csv(comparison_output, index=False, encoding='utf-8')
        print(f"   Total verdicts with significant changes: {df_changes['verdict_id'].nunique()}")
    else:
        print("ℹ️  No facts found with changes > 2 words")
    
    if dry_run:
        print("🔍 DRY RUN MODE: Database was NOT updated. Only comparison CSV generated.")
        return
    
    # 4. Actually update the database (only if not dry run)
    print("⚙️  Updating rows in Similarity Database...")
    
    # Update facts_1
    new_values_1 = df_sim[sim_verdict1_col].map(facts_map)
    if sim_facts1_col in df_sim.columns:
        original_1 = df_sim[sim_facts1_col]
        df_sim[sim_facts1_col] = new_values_1.fillna(original_1)
    else:
        df_sim[sim_facts1_col] = new_values_1.fillna("")

    # Update facts_2
    new_values_2 = df_sim[sim_verdict2_col].map(facts_map)
    if sim_facts2_col in df_sim.columns:
        original_2 = df_sim[sim_facts2_col]
        df_sim[sim_facts2_col] = new_values_2.fillna(original_2)
    else:
        df_sim[sim_facts2_col] = new_values_2.fillna("")

    # 5. Save
    print(f"💾 Saving updated file to: {output_path}")
    df_sim.to_csv(output_path, index=False)
    print("✅ Done!")

# ==========================================
# RUN CONFIGURATION
# ==========================================
if __name__ == "__main__":
    # 1. Path to your existing Similarity Database (the pairs file)
    SIM_DB_PATH = base_path+"similarity_database_with_indicment_facts.csv"
    
    # 2. Path to the NEW CSV with the correct facts (the master list)
    NEW_FACTS_PATH = base_path+'gpt/processed_verdicts_with_gpt.csv'
    
    # 3. Output file name
    OUTPUT_PATH = base_path+"similarity_database_with_indicment_facts.csv"    
    # 4. Column Name Mapping (Check your CSV headers!)
    # What are the column names in the NEW file?
    NEW_ID_COLUMN = "verdict"          # e.g., 'id', 'verdict_id', 'CaseID'
    NEW_TEXT_COLUMN = "extracted_gpt_facts"  # e.g., 'facts', 'text', 'indictment_facts'

    update_similarity_database(
        similarity_db_path=SIM_DB_PATH,
        new_facts_csv_path=NEW_FACTS_PATH,
        output_path=OUTPUT_PATH,
        new_id_col=NEW_ID_COLUMN,
        new_text_col=NEW_TEXT_COLUMN,
        dry_run=DRY_RUN
    )