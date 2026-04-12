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


DRY_RUN = False  # Set to False to process files and call the GPT API
PROCESS_ONLY_TARGET_VERDICTS = True  # Set to True to process only verdicts from target.csv
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

# base_path='/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/data/drugs_3k/docx'
# csv_directory ='/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/data/drugs_3k/verdict_csv'
# out_dir = '/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/data/drugs_3k'

os.makedirs(out_dir, exist_ok=True)

output_file = os.path.join(out_dir, "processed_verdicts_with_gpt.csv")
failed_file = os.path.join(out_dir, "failed_verdicts.csv")

# ========== Load Target CSV and Get Unique Verdicts (if flag is set) ==========
unique_verdicts = None
if PROCESS_ONLY_TARGET_VERDICTS:
    target_csv_path = os.path.join(base_path, "target.csv")
    if os.path.exists(target_csv_path):
        target_df = pd.read_csv(target_csv_path)
        # Extract unique verdicts from both verdict_1 and verdict_2 columns
        unique_verdicts = set(target_df['verdict_1'].astype(str).str.strip().unique()) | set(target_df['verdict_2'].astype(str).str.strip().unique())
        print(f"✅ Processing only {len(unique_verdicts)} unique verdicts from target CSV")
    else:
        print(f"⚠️ Target CSV not found at {target_csv_path}, processing all verdicts")
else:
    print(f"ℹ️ Processing all verdicts (PROCESS_ONLY_TARGET_VERDICTS is False)")

# ========== Pattern Definitions ==========
START_PARTS = ["עובדותם", "כללי", "כתב האישום", "האישום", "אישום", "רקע", "גזר", "דין", "פסק","מבוא","הרשעת" ,"בעניינו","עבירות","הורשע","עובדות","השתלשלות", "ג ז ר",  "ד י ן","פתח דבר","פתח"]
END_PARTS = ["טענות", "עמדת", "תסקיר","תסקירי", "שירות", "מבחן", "דיון", "התסקיר","טיעוני", "הצדדים", "צדדים", "והכרעה",  "ראיות","החלטה"]

# ========== Helper Functions ==========
def extract_indictment_facts(df):
    if df.empty or "part" not in df.columns or "text" not in df.columns:
        return "❌ No indictment facts found", None, None, 0

    df["part"] = df["part"].astype(str).str.strip()
    
    # Exclude irrelevant parts from being considered as start parts
    EXCLUDED_START_PARTS = ["כתבי עת", "חקיקה שאוזכרה", "חקיקה", "ציטוטים", "מקורות"]
    
    start_row = df[df["part"].str.contains('|'.join(START_PARTS), case=False, na=False, regex=True)]
    # Filter out excluded parts
    if not start_row.empty:
        excluded_mask = start_row["part"].str.contains('|'.join(EXCLUDED_START_PARTS), case=False, na=False, regex=True)
        start_row = start_row[~excluded_mask]
    
    if start_row.empty:
        # If no valid start part found, try to find any part that contains indictment keywords in its text
        # This is a fallback for cases where the part name doesn't match START_PARTS but the content does
        for idx, row in df.iterrows():
            text_content = str(row.get("text", "")).strip() if pd.notna(row.get("text")) else ""
            if text_content:
                text_lower = text_content.casefold()
                # Check if this text contains indictment keywords
                indictment_keywords = ["הורשע", "הרשענו", "מצאנו להרשיעו", "כתב אישום", "הנאשם הורשע"]
                if any(keyword in text_lower for keyword in indictment_keywords):
                    start_idx = idx
                    start_part_name = df.loc[idx, "part"]
                    normalized_start_part = re.sub(r"\s+", " ", str(start_part_name).strip().casefold())
                    has_start = True
                    break
        else:
            # No valid start found even after fallback search
            start_idx = 0
            start_part_name = "❌ No start found (use index 0)"
            normalized_start_part = None
            has_start = False
    else:
        start_idx = start_row.index.min()
        start_part_name = df.loc[start_idx, "part"]
        normalized_start_part = re.sub(r"\s+", " ", str(start_part_name).strip().casefold())
        has_start = True

    end_mask = (
        (df.index > start_idx) &
        (df["part"].str.contains('|'.join(END_PARTS), case=False, na=False, regex=True))
    )
    end_row = df.loc[end_mask]

    end_candidates = end_row.index.tolist()
    valid_end_idx = None

    for candidate_idx in end_candidates:
        candidate_part = str(df.loc[candidate_idx, "part"]).strip()
        
        # FIRST: Check if this candidate matches START_PARTS - if so, skip it immediately
        # Any part that matches START_PARTS cannot be an end part
        if candidate_part:
            # Use the same regex pattern matching as used for finding start parts
            candidate_series = pd.Series([candidate_part])
            matches_start_pattern = candidate_series.str.contains('|'.join(START_PARTS), case=False, na=False, regex=True).iloc[0]
            if matches_start_pattern:
                # If it matches START_PARTS, it cannot be an end part - skip it
                # (This handles cases like "גזר דין...דיון" which shouldn't be an end part)
                continue  # Skip any part that matches START_PARTS
        
        # SECOND: Check if it's the same as or contains the start part
        if has_start:
            normalized_candidate = re.sub(r"\s+", " ", candidate_part.casefold())
            if normalized_candidate == normalized_start_part:
                continue
            # If end part includes the start part, skip it
            if normalized_start_part and normalized_start_part in normalized_candidate:
                continue
        
        # If we got here, this is a valid end part
        valid_end_idx = candidate_idx
        break

    if valid_end_idx is not None:
        end_idx = valid_end_idx
        end_part_name = df.loc[end_idx, "part"]
    else:
        end_idx = len(df)
        end_part_name = "❌ No end found (used full text)"

    # Count parts between start and end (inclusive of start, exclusive of end)
    parts_count = end_idx - start_idx
    
    # Extract text grouped by part with part names as headers
    extracted_sections = []
    current_part = None
    current_text_parts = []
    
    for idx in range(start_idx, end_idx):
        row = df.loc[idx]
        part_name = str(row["part"]).strip()
        text_content = str(row["text"]).strip() if pd.notna(row["text"]) else ""
        
        if not text_content:
            continue
            
        # If this is a new part, save the previous part and start a new one
        if part_name != current_part:
            if current_part is not None and current_text_parts:
                # Add the previous part with its header
                extracted_sections.append(f"{current_part}:")
                extracted_sections.append("\n".join(current_text_parts))
                extracted_sections.append("")  # Empty line between parts
            
            current_part = part_name
            current_text_parts = [text_content]
        else:
            # Same part, just append the text
            current_text_parts.append(text_content)
    
    # Don't forget the last part
    if current_part is not None and current_text_parts:
        extracted_sections.append(f"{current_part}:")
        extracted_sections.append("\n".join(current_text_parts))
    
    extracted_text = "\n".join(extracted_sections)
    
    # Validate that the extracted text actually contains indictment-related content
    # If it only contains citations, legislation, or other non-indictment content, return empty
    if extracted_text:
        extracted_text_lower = extracted_text.casefold()
        # Keywords that indicate this is actually an indictment section
        indictment_keywords = [
            "הורשע", "הרשענו", "מצאנו להרשיעו", "כתב אישום", "כתב האישום",
            "על פי הודאתו", "על פי הודאת", "הודה", "הנאשם הורשע", "הנאשם הודה",
            "על פי הנטען בכתב האישום", "על פי עובדות הכרעת הדין", "על פי עובדות כתב האישום",
            "על פי הממצאים שנקבעו בהכרעת הדין", "בכתב האישום", "בכתב אישום",
            "הסדר טיעון", "בעבירות", "לפי סעיף", "לפי סעיפים"
        ]
        
        # Check if text contains any indictment keywords
        has_indictment_content = any(keyword in extracted_text_lower for keyword in indictment_keywords)
        
        # If the text is mostly citations/legislation (contains many law references but no indictment)
        # or if it's very short and doesn't contain indictment keywords, it's probably not an indictment
        if not has_indictment_content:
            # Check if it's mostly citations/legislation
            citation_indicators = ["חוק העונשין", "פקודת", "תקנות", "ע\"פ", "ע\"א", "ע\"מ", "ע\"ב"]
            has_citations = any(indicator in extracted_text_lower for indicator in citation_indicators)
            
            # If it has citations but no indictment content, it's probably not an indictment section
            if has_citations and len(extracted_text.split()) < 50:  # Short text with citations but no indictment
                return "❌ No indictment facts found", start_part_name, end_part_name, parts_count
    
    return extracted_text.strip() if extracted_text else "❌ No indictment facts found", start_part_name, end_part_name, parts_count


def extract_facts_with_gpt(text):
    """
    Sends extracted text to GPT API and extracts specific facts.
    """
    if text == "❌ No indictment facts found" :
        return "GPT extraction error"


    prompt = f"""
תפקידך הוא לחלץ מידע משפטי מתוך טקסט של גזר דין.
המטרה שלך היא למצוא את "הסיפור העובדתי" - בגין מה הורשע הנאשם ומה בדיוק קרה שם.

עליך לחלץ שני חלקים:
1. **פסקת האישום/ההרשעה**: המשפט הפורמלי שקובע במה הנאשם הורשע (סעיפי חוק, סוג העבירה, הודאה/הכחשה).
2. **תיאור העובדות**: הסיפור המלא של המקרה (מה קרה, מתי, איפה, מי המעורבים).

הנחיות לביצוע:
1. חפש עוגנים כמו: "הנאשם הורשע", "על פי עובדות כתב האישום", "כתב האישום המתוקן", "העובדות בהן הודה".
2. אם הטקסט מכיל תיאור עובדתי (סיפור המעשה) מיד לאחר ההרשעה - העתק את כולו.
3. **אל תסכם**. העתק את הטקסט המקורי מילה במילה (Copy-Paste).
4. **מתי לעצור?** הפסק להעתיק כאשר הטקסט עובר לנושאים אחרים כגון: "תסקיר שירות המבחן", "טיעונים לעונש", "ראיות לעונש", "דיון והכרעה" או ניתוח משפטי.


כעת, עבד את הטקסט הבא:
  {text}

    החזר את הפלט בפורמט הבא בלבד:
    <פסקת כתב האישום>

    <פסקת עובדות כתב האישום>


    """

    response = client.chat.completions.create(
        model="gpt-5.2", 
        messages=[
            {"role": "system", "content": "אתה מודל בינה מלאכותית שתפקידו לחלץ עובדות מכתבי אישום בטקסטים משפטיים בעברית, מבלי לפרש, לסכם או לשנות את הנוסח המקורי."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()
     

# ========== Load Existing Data ==========
if os.path.exists(output_file):
    processed_df = pd.read_csv(output_file)
    # Create backup for comparison (only if backup doesn't exist)
    backup_file = output_file.replace(".csv", "_old.csv")
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(output_file, backup_file)
        print(f"📋 Created backup of existing results: {backup_file}")
        print(f"   (This backup will be used for comparison after processing)")
else:
    processed_df = pd.DataFrame(columns=["verdict", "extracted_facts", "extracted_gpt_facts","start_part","end_part", "parts_count"])

# ========== Process Files ==========
file_list = [f for f in os.listdir(csv_directory) if f.endswith(".csv")]
processed_df["verdict"] = processed_df["verdict"].astype(str).str.strip()
failed_verdicts = []
unique_start_end_pairs = set()
parts_count_stats = []  # Track parts count for each verdict

for filename in tqdm(file_list, desc="Processing verdicts"):
    file_path = os.path.join(csv_directory, filename)
    try:
        df = pd.read_csv(file_path)
        verdict_id = str(df["verdict"].iloc[0]).strip()
        
        # ========== Skip if verdict is not in target CSV ==========
        if unique_verdicts is not None and verdict_id not in unique_verdicts:
            continue

        # Extract
        extracted_facts, start_part, end_part, parts_count = extract_indictment_facts(df)
        extracted_facts_normalized = extracted_facts.strip() if isinstance(extracted_facts, str) else ""
        start_label = start_part.strip() if isinstance(start_part, str) else str(start_part)
        end_label = end_part.strip() if isinstance(end_part, str) else str(end_part)
        unique_start_end_pairs.add((start_label, end_label))
        
        # Track parts count statistics
        parts_count_stats.append({
            "verdict": verdict_id,
            "parts_count": parts_count,
            "start_part": start_label,
            "end_part": end_label
        })

        # Check if already processed and if extraction is valid (not empty and not too short)
        # Check if GPT extraction is valid (not empty and not too short)
        def is_valid_extraction(gpt_text):
            if pd.isna(gpt_text) or not isinstance(gpt_text, str):
                return False
            gpt_text = gpt_text.strip()
            if gpt_text == "" or gpt_text == "GPT extraction error":
                return False
            # # Check if it's too short (≤2 sentences)
            # sentences = [s.strip() for s in re.split(r'[.!?]', gpt_text) if s.strip()]
            # if len(sentences) <= 1:
            #     return False
            return True

        if DRY_RUN:
            continue

        # Check if already processed and if extraction is valid
        existing_rows = processed_df[processed_df["verdict"] == verdict_id]
        if not existing_rows.empty:
            existing_gpt = existing_rows["extracted_gpt_facts"].iloc[0]
            # If extraction is valid, skip GPT call and use existing result
            if is_valid_extraction(existing_gpt):
                print(f"⏭️ Skipping GPT for verdict {verdict_id} (valid extraction exists)")
                continue

        # Only run GPT extraction if no valid extraction exists
        extracted_gpt_facts = extract_facts_with_gpt(extracted_facts)
        

        # Save or update
        existing_rows = processed_df[processed_df["verdict"] == verdict_id]
        if not existing_rows.empty:
            # Update existing row
            idx = existing_rows.index[0]
            processed_df.at[idx, "extracted_facts"] = extracted_facts
            processed_df.at[idx, "extracted_gpt_facts"] = extracted_gpt_facts
            processed_df.at[idx, "start_part"] = start_part
            processed_df.at[idx, "end_part"] = end_part
            processed_df.at[idx, "parts_count"] = parts_count
        else:
            # Create new row
            new_row = pd.DataFrame([{
                "verdict": verdict_id,
                "extracted_facts": extracted_facts,
                "extracted_gpt_facts": extracted_gpt_facts,
                "start_part": start_part,
                "end_part": end_part,
                "parts_count": parts_count
            }])
            processed_df = pd.concat([processed_df, new_row], ignore_index=True)
        processed_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        
        time.sleep(1)  # avoid rate limits

    except Exception as e:
        failed_verdicts.append({"verdict": filename, "reason": str(e)})

    gc.collect()

if DRY_RUN:
    print(f"Found {len(unique_start_end_pairs)} unique start/end combinations.")
    for start_label, end_label in sorted(unique_start_end_pairs, key=lambda pair: (pair[0], pair[1])):
        print(f"START: {start_label} | END: {end_label}")

# ========== Save Failures ==========
if failed_verdicts:
    pd.DataFrame(failed_verdicts).to_csv(failed_file, index=False, encoding="utf-8-sig")

# ========== Empty and Short Extractions Analysis ==========
print("\n" + "="*60)
print("EMPTY AND SHORT EXTRACTIONS ANALYSIS")
print("="*60)

if not processed_df.empty and "extracted_gpt_facts" in processed_df.columns and "extracted_facts" in processed_df.columns:
    # Find empty extractions
    def is_empty(text):
        if pd.isna(text) or not isinstance(text, str):
            return True
        return text.strip() == "" or text.strip() == "GPT extraction error"
    
    # Find short extractions (up to 2 sentences)
    def is_short(text):
        if pd.isna(text) or not isinstance(text, str) or is_empty(text):
            return False
        # Count sentences (split by period, exclamation, question mark)
        sentences = [s.strip() for s in re.split(r'[.!?]', text.strip()) if s.strip()]
        return len(sentences) <= 2
    
    processed_df["is_empty"] = processed_df["extracted_gpt_facts"].apply(is_empty)
    processed_df["is_short"] = processed_df["extracted_gpt_facts"].apply(is_short)
    
    empty_extractions = processed_df[processed_df["is_empty"]].copy()
    short_extractions = processed_df[processed_df["is_short"] & ~processed_df["is_empty"]].copy()
    
    empty_count = len(empty_extractions)
    short_count = len(short_extractions)
    
    print(f"\n📊 Empty Extractions: {empty_count} out of {len(processed_df)} total")
    print(f"📊 Short Extractions (≤2 sentences): {short_count} out of {len(processed_df)} total")
    
    all_problematic = pd.concat([empty_extractions, short_extractions]).drop_duplicates(subset=['verdict'])
    
    if len(all_problematic) > 0:
        print(f"\n📝 Input Text for Empty/Short Extractions:")
        for idx, row in all_problematic.iterrows():
            verdict_id = row['verdict']
            input_text = str(row['extracted_facts']) if pd.notna(row['extracted_facts']) else ""
            gpt_output = str(row['extracted_gpt_facts']) if pd.notna(row['extracted_gpt_facts']) else ""
            
            is_empty_case = row['is_empty']
            is_short_case = row['is_short'] if not is_empty_case else False
            
            # Show which parts were included
            if "start_part" in row and "end_part" in row:
                start_part = row['start_part']
                end_part = row['end_part']
                parts_count = row.get('parts_count', 'N/A')
                print(f"\n  Verdict: {verdict_id}")
                print(f"  Start part: {start_part}")
                print(f"  End part: {end_part}")
                # print(f"  Parts included: {parts_count}")
                if is_empty_case:
                    print(f"  Status: EMPTY")
                # elif is_short_case:
                #     print(f"  Status: SHORT (≤2 sentences)")
            
            # Show part names from the input text (lines ending with :)
            part_names = [line.strip() for line in input_text.split("\n") if line.strip().endswith(":")]
            if part_names:
                print(f"  Part names in input: {', '.join(part_names[:5])}")
            
            if gpt_output:
                print(f"  GPT output: {gpt_output[:500]}")
            else:
                print(f"  GPT output: (empty)")
            
            print(f"  Input text: {input_text[:500]}")
    else:
        print("\n✅ No empty or short extractions found!")
else:
    print("⚠️ No processed data available for analysis.")

