#!/usr/bin/env python3
"""
Few-shot GPT classifier + extractor:
1. Filter by regex OR required parts from CSV
2. Use GPT with few-shot examples to identify the sentence where judge declares punishment range
3. Extract the range and convert to months
4. Process all cases from target.csv files in drugs and weapon directories

Usage:
    python 3_extract_sentencing_range.py --csv-dir /path/to/verdict_csv --output-dir /path/to/output
    
    Or without arguments to use default hardcoded paths.
"""
import os
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
import re
import sys
import signal
import argparse
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

# Timeout handler for slow file reads (OneDrive sync issues)
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("File read timed out")

def read_csv_with_timeout(file_path, timeout_seconds=30):
    """Read CSV with timeout to handle OneDrive sync delays"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        df = pd.read_csv(file_path)
        signal.alarm(0)  # Cancel the alarm
        return df
    except TimeoutError:
        signal.alarm(0)
        return None
    except Exception as e:
        signal.alarm(0)
        return None

# Initialize OpenAI client
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise SystemExit('Please set OPENAI_API_KEY environment variable before running.')
client = OpenAI(api_key=API_KEY)

# Default CSV directories - used when no command line arguments provided
DEFAULT_CSV_DIRS = [
    # '/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/weapon/verdict_csv',  # Already done
    # '/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/new_try/drugs/verdict_csv',  # Already done
    '/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/innovation_task/5k/verdict_csv',  # 5k dataset - NEED TO RUN
]

# Regex pattern and required parts
PUNISHMENT_PATTERN = "(מתח. ה?עו?ני?ש|מתחם)"
REQUIRED_PARTS = [
    "מתחמי ענישה", "אחידות בענישה", "מתחם הענישה", "מתחם ענישה",
    "מתחמי הענישה", "מתחם העונש", "מתחם עונש"
]

# Load ground truth files to get verdicts to process (optional)
GT_FILES = [
    '/Users/liorb/Library/CloudStorage/OneDrive-post.bgu.ac.il/Thesis!!!/innovation_task/features_gt_weapom - מתחם ענישה - שופט+ נרמול.csv'
]

# Few-shot examples for GPT
FEW_SHOT_EXAMPLES = """
מטרה: לזהות את המשפט שבו השופט/ת מכריז/ת על מתחם הענישה בתיק הנוכחי, ולחלץ את המתחם.

דוגמה 1:
טקסט: "לאחר ששקלתי את כל הנסיבות, אני קובע כי מתחם העונש ההולם נע בין 12 חודשי מאסר ל-24 חודשי מאסר."
משפט רלוונטי: "אני קובע כי מתחם העונש ההולם נע בין 12 חודשי מאסר ל-24 חודשי מאסר"
מתחם: 12 - 24 חודשים

דוגמה 2:
טקסט: "הנאשם הורשע בעבירה של החזקת נשק. הוא ירצה 6 חודשי מאסר."
משפט רלוונטי: אין
מתחם: אין

דוגמה 3:
טקסט: "לטענת המדינה מתחם העונש ההולם נמצא בין שנתיים עד ארבע שנות מאסר. מכל האמור הגעתי למסקנה שמתחם העונש ההולם נמצא בין 9 חודשי מאסר בפועל עד 3 שנות מאסר בפועל."
משפט רלוונטי: "מכל האמור הגעתי למסקנה שמתחם העונש ההולם נמצא בין 9 חודשי מאסר בפועל עד 3 שנות מאסר בפועל"
מתחם: 9 - 36 חודשים (3 שנים = 36 חודשים)

דוגמה 4:
טקסט: "ע''פ 2482/22 מדינת ישראל נ' קדורה: נקבע מתחם הנע בין 10-36 חודשי מאסר. לאחר בחינת כלל השיקולים מצאתי כי יש לקבוע בנסיבות מקרה זה מתחם ענישה של 24-40 חודשי מאסר בפועל."
משפט רלוונטי: "לאחר בחינת כלל השיקולים מצאתי כי יש לקבוע בנסיבות מקרה זה מתחם ענישה של 24-40 חודשי מאסר בפועל"
מתחם: 24 - 40 חודשים
הערה: המתחם מתיק אחר (ע"פ 2482/22) נדחה - רק המתחם שנקבע בתיק הנוכחי רלוונטי.

דוגמה 5 (מתחמים מרובים מאותו תיק):
טקסט: "אני קובעת כי מתחם הענישה באישום הראשון נע בין שבע לבין עשר שנות מאסר בפועל; לאישום השני מאסר בפועל בין 36 לבין 60 חודשים."
משפט רלוונטי: "אני קובעת כי מתחם הענישה באישום הראשון נע בין שבע לבין עשר שנות מאסר בפועל; לאישום השני מאסר בפועל בין 36 לבין 60 חודשים"
מתחם: 84 - 120 חודשים (7 שנים=84, 10 שנים=120, חישוב: min(84,36)=36, max(120,60)=120, אבל אם יש מתחמים נפרדים - קח את המקסימום של כל הנמוכים: max(84,36)=84, ואת המקסימום של כל הגבוהים: max(120,60)=120)

דוגמה 6 (מתחמים מתיקים שונים - לקחת רק את המתחם העיקרי):
טקסט: "לאור כל האמור אני סבורה שמתחמי הענישה שיש לקבוע במקרה זה הם כדלקמן: בתיק הנשק – מתחם כולל לשני האישומים הנע בין שתיים לארבע שנות מאסר. בתיק הסמים – מתחם כולל שבין שנה לשנתיים מאסר. בתיק הגניבה – מתחם שבין עונש שאינו כולל מאסר בפועל לבין מאסר קצר."
משפט רלוונטי: "לאור כל האמור אני סבורה שמתחמי הענישה שיש לקבוע במקרה זה הם כדלקמן: בתיק הנשק – מתחם כולל לשני האישומים הנע בין שתיים לארבע שנות מאסר"
מתחם: 24 - 48 חודשים (2-4 שנים, רק מתחם הנשק - המתחם העיקרי)
הערה: אם יש מתחמים מתיקים שונים, לקחת רק את המתחם העיקרי. אם יש מתחמים מרובים מאותו תיק, לקחת את המקסימום של כל הערכים הנמוכים ואת המקסימום של כל הערכים הגבוהים.

דוגמה 7 (חשוב: לא להתבלבל בין המתחם המעוגן בחוק למתחם שנקבע בפועל - רק אם שניהם מופיעים):
טקסט: "חומרת הפגיעה היא בינונית עד גבוהה שכן מדובר בשני כלי נשק ואחד מהם הינו רובה. מתחם העונש הינו בין 30 ל- 60 חודשי מאסר ובתיק זה אין כל נסיבה המצדיקה לסטות מהמתחם. בהתאם לתיקון 113 לחוק העונשין (סעיף 40 יג'), סבורני כי מתחם העונש ההולם הינו החל מ- 18 ועד 42 חודשי מאסר."
משפט רלוונטי: "בהתאם לתיקון 113 לחוק העונשין (סעיף 40 יג'), סבורני כי מתחם העונש ההולם הינו החל מ- 18 ועד 42 חודשי מאסר"
מתחם: 18 - 42 חודשים
הערה חשובה: רק אם שני סוגי מתחמים מופיעים יחד:
1. המתחם המעוגן בחוק/הסטטוטורי (כמו "מתחם העונש הינו בין 30 ל- 60 חודשי מאסר") - זה המתחם הכללי בחוק
2. המתחם שנקבע בפועל בתיק הנוכחי על ידי השופט (כמו "סבורני כי מתחם העונש ההולם הינו החל מ- 18 ועד 42 חודשי מאסר") - זה המתחם הרלוונטי
אם שניהם מופיעים - תמיד לקחת את המתחם שנקבע בפועל על ידי השופט בתיק הנוכחי, לא את המתחם הסטטוטורי הכללי!
אם רק מתחם אחד מופיע - לקחת אותו.

דוגמה 7ב (מקרה שבו יש רק מתחם אחד מפסיקה - לקחת אותו):
טקסט: "המאשימה מציינת כי מדיניות הענישה הנהוגה בעבירה זו עומדת על מאסר בפועל. בהקשר זה מדגישה כי המלצת שירות המבחן בפרט כשמדובר בנאשם בעל עבר פלילי חורגת ממתחם הענישה ובתוך כך מפנה לפסיקה הבאה ממנה עולה כי מתחם הענישה נע בין 12- 24 חודשי מאסר בפועל."
משפט רלוונטי: "ממנה עולה כי מתחם הענישה נע בין 12- 24 חודשי מאסר בפועל"
מתחם: 12 - 24 חודשים
הערה: במקרה זה יש רק מתחם אחד המוזכר (מפסיקה), ולכן הוא המתחם הרלוונטי - גם אם הוא לא נקבע ישירות על ידי השופט בתיק הנוכחי.
"""

def split_into_sentences(text: str) -> list:
    """Split text on sentence boundaries."""
    if not isinstance(text, str):
        return []
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def process_text_for_gpt(text: str) -> list:
    """
    Process text: split into sentences, check regex/required parts, add context.
    """
    if not text or not text.strip():
        return []
    
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    processed_texts = []
    seen_combined = set()  # Track seen combinations to avoid duplicates
    
    for i, sentence in enumerate(sentences):
        matches_regex = sentence and re.search(PUNISHMENT_PATTERN, sentence)
        matches_required_part = sentence and any(part in sentence for part in REQUIRED_PARTS)
        # Also check if sentence contains numeric range (like "17-10 חודשים" or "10-20 חודשי מאסר")
        matches_numeric_range = sentence and re.search(r'\d+\s*[-–]\s*\d+\s*(חודש|חודשים|חודשי|שנה|שנים|שנת)', sentence)
        
        if matches_regex or matches_required_part or matches_numeric_range:
            if i > 0:
                combined = sentences[i-1] + ' ' + sentence
                # Normalize for comparison
                normalized = ' '.join(combined.split())
                if normalized not in seen_combined:
                    seen_combined.add(normalized)
                    processed_texts.append(combined)
            else:
                if i < len(sentences) - 1:
                    combined = sentence + ' ' + sentences[i+1]
                    normalized = ' '.join(combined.split())
                    if normalized not in seen_combined:
                        seen_combined.add(normalized)
                        processed_texts.append(combined)
                else:
                    normalized = ' '.join(sentence.split())
                    if normalized not in seen_combined:
                        seen_combined.add(normalized)
                        processed_texts.append(sentence)
    
    return processed_texts

def gpt_classify_and_extract(text: str) -> dict:
    """
    Use GPT to identify the sentence where the judge declares the punishment range for the current case,
    and extract the range.
    Returns: {'sentence': str or None, 'range': (low, high) or None, 'confidence': 'HIGH'/'MEDIUM'/'LOW'}
    """
    prompt = f"""{FEW_SHOT_EXAMPLES}

כעת נתח את הטקסט הבא:

טקסט: "{text}"

מטרה: לזהות את המשפט שבו השופט/ת מכריז/ת על מתחם הענישה בתיק הנוכחי.

כללים:
1. מחפשים רק משפט שבו השופט/ת מכריז/ת ישירות על מתחם ענישה בתיק הנוכחי
2. דחה:
   - מתחמים מתיקים אחרים (אם מוזכר תיק אחר: ע"פ, ת"פ, עפ"ג, רע"פ וכו')
   - מתחמים של "לטענת המדינה", "ב"כ המאשימה טוען", "ב"כ הנאשם מבקש"
   - מתחמים ללא מאסר בפועל (כמו "עונש שאינו כולל מאסר בפועל")
   - המתחם הסטטוטורי/המעוגן בחוק - רק אם יש גם מתחם שנקבע בפועל על ידי השופט בתיק הנוכחי. אם יש רק מתחם אחד (גם אם הוא מפסיקה/תקדים) - לקחת אותו.
3. משפטים רלוונטיים כוללים: "אני קובע", "הגעתי למסקנה", "מצאתי כי יש לקבוע", "סבורני כי מתחם העונש ההולם", "מתחם ענישה", "אני סבורה שמתחמי הענישה"
4. אם יש מתחמים מרובים מאותו תיק:
   - חשוב מאוד: קח את המקסימום של כל הערכים הנמוכים (לא המינימום!) ואת המקסימום של כל הערכים הגבוהים
   - דוגמה: אם יש מתחם 10-24 ומתחם 18-48 → התוצאה היא max(10,18) עד max(24,48) = 18-48 (לא 10-48!)
   - אם יש מתחמים מתיקים שונים (נשק, סמים, גניבה וכו') - לקחת רק את המתחם העיקרי (לרוב הנשק)
5. המר שנים לחודשים: שנה אחת = 12 חודשים
6. חשוב מאוד: הבחן בין המתחם הסטטוטורי (מעוגן בחוק) למתחם שנקבע בפועל - רק אם שניהם מופיעים יחד! אם יש רק מתחם אחד מסוג זה, לקחת אותו. אם יש גם מתחם סטטוטורי וגם מתחם שנקבע בפועל - תמיד לקחת את המתחם שנקבע בפועל על ידי השופט בתיק הנוכחי!

השב בפורמט המדויק הבא:
משפט רלוונטי: [המשפט שבו השופט מכריז על המתחם, או "אין" אם לא נמצא]
מתחם: [X - Y חודשים או "אין"]
רמת ביטחון: [גבוהה/בינונית/נמוכה]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Low temperature for consistency
        )
        gpt_output = response.choices[0].message.content.strip()
        
        # Parse response (Hebrew)
        sentence = None
        range_str = None
        confidence = None
        
        for line in gpt_output.split('\n'):
            if 'משפט רלוונטי:' in line or 'Relevant sentence:' in line:
                if 'משפט רלוונטי:' in line:
                    sentence = line.split('משפט רלוונטי:')[1].strip()
                else:
                    sentence = line.split('Relevant sentence:')[1].strip()
                # Remove quotes if present
                if sentence.startswith('"') and sentence.endswith('"'):
                    sentence = sentence[1:-1]
            elif 'מתחם:' in line or 'Range:' in line:
                if 'מתחם:' in line:
                    range_str = line.split('מתחם:')[1].strip()
                else:
                    range_str = line.split('Range:')[1].strip()
            elif 'רמת ביטחון:' in line or 'Confidence:' in line:
                if 'רמת ביטחון:' in line:
                    confidence = line.split('רמת ביטחון:')[1].strip()
                else:
                    confidence = line.split('Confidence:')[1].strip()
        
        # Extract range
        low, high = None, None
        if range_str and 'אין' not in range_str and 'None' not in range_str:
            match = re.search(r'(\d+)\s*[-–]\s*(\d+)', range_str)
            if match:
                low, high = int(match.group(1)), int(match.group(2))
        
        # Determine classification based on whether we found a range
        classification = 'POSITIVE' if (low and high) else 'NEGATIVE'
        
        return {
            'classification': classification,
            'sentence': sentence if sentence and 'אין' not in sentence and 'None' not in sentence else None,
            'range': (low, high) if low and high else None,
            'range_str': range_str,
            'confidence': confidence,
            'gpt_output': gpt_output
        }
    except Exception as e:
        return {
            'classification': 'ERROR',
            'sentence': None,
            'range': None,
            'range_str': None,
            'confidence': None,
            'gpt_output': f'Error: {str(e)}'
        }

def process_directories(csv_dirs, output_dir=None, filter_by_target=False, target_verdicts=None):
    """
    Process CSV directories and extract sentencing ranges.
    
    Args:
        csv_dirs: List of CSV directories to process
        output_dir: Output directory for results (default: current directory)
        filter_by_target: Whether to filter by target verdicts
        target_verdicts: Set of target verdict names to process
    """
    if target_verdicts is None:
        target_verdicts = set()
    
    # Collect all filtered texts per verdict per directory
    # Dictionary: csv_dir -> verdict -> list of texts (will aggregate them)
    verdict_texts_by_dir = {}

    print('Filtering texts by regex and required parts...')
    print('='*60)

    for csv_dir in csv_dirs:
        if not os.path.exists(csv_dir):
            print(f'⚠️  Directory not found: {csv_dir}')
            continue
        
        # Initialize dictionary for this directory
        if csv_dir not in verdict_texts_by_dir:
            verdict_texts_by_dir[csv_dir] = {}
        
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        print(f'\n📂 Processing {len(csv_files)} files from {os.path.basename(csv_dir)}')
        
        skipped_count = 0
        for i, csv_file in enumerate(tqdm(csv_files, desc=f"Reading {os.path.basename(csv_dir)}")):
            verdict_name = csv_file.replace('.csv', '')
            
            # Filter by target verdicts if filter_by_target is enabled
            if filter_by_target and target_verdicts and verdict_name not in target_verdicts:
                skipped_count += 1
                continue
            
            # Initialize list for this verdict if not exists
            if verdict_name not in verdict_texts_by_dir[csv_dir]:
                verdict_texts_by_dir[csv_dir][verdict_name] = []
        
            file_path = os.path.join(csv_dir, csv_file)
            df = read_csv_with_timeout(file_path, timeout_seconds=30)
            if df is None:
                continue
            
            for _, row in df.iterrows():
                # Combine text from multiple columns
                text_parts = []
                part_value = None
                sentence_text = None
                
                for col in ['text', 'part', 'content', 'sentence', 'paragraph']:
                    if col in row and pd.notna(row[col]):
                        part_text = str(row[col]).strip()
                        if part_text:
                            if col == 'part':
                                part_value = part_text
                            elif col in ['text', 'content', 'sentence', 'paragraph']:
                                sentence_text = part_text
                            text_parts.append(part_text)
                
                if not text_parts:
                    for col in df.columns:
                        if col not in ['verdict', 'case', 'id'] and pd.notna(row[col]):
                            part_text = str(row[col]).strip()
                            if part_text and len(part_text) > 10:
                                if col == 'part':
                                    part_value = part_text
                                else:
                                    sentence_text = part_text
                                text_parts.append(part_text)
                
                combined_text = ' '.join(text_parts) if text_parts else ''
                if not combined_text:
                    continue
                
                # Check if part matches required parts
                part_matches_required = part_value and any(required_part in part_value for required_part in REQUIRED_PARTS)
                
                # Priority: If part matches required parts, take raw text from CSV (NO sentence splitting)
                # Otherwise, use regex matching with sentence splitting
                if part_matches_required:
                    # Take the raw combined text directly from CSV (no processing/splitting)
                    # This includes both part and sentence as-is
                    processed_texts = [combined_text]
                else:
                    # Process normally (filters by regex) - splits into sentences
                    processed_texts = process_text_for_gpt(combined_text)
                
                # Deduplicate
                all_processed = processed_texts
                
                # Deduplicate by normalizing and comparing
                seen_normalized = set()
                unique_processed = []
                for text in all_processed:
                    if text:
                        normalized = ' '.join(text.split())
                        if normalized not in seen_normalized:
                            seen_normalized.add(normalized)
                            unique_processed.append(text)
                
                # Add all unique processed texts to this verdict's collection
                for processed_text in unique_processed:
                    if processed_text:
                        verdict_texts_by_dir[csv_dir][verdict_name].append(processed_text)
        
        if filter_by_target and skipped_count > 0:
            print(f'  ⚠️  Skipped {skipped_count} verdicts not in target list')

    # Process each directory separately
    print(f'\n{"="*60}')
    total_verdicts = sum(len(verdicts) for verdicts in verdict_texts_by_dir.values())
    total_texts = sum(sum(len(texts) for texts in verdicts.values()) for verdicts in verdict_texts_by_dir.values())
    print(f'Total filtered texts: {total_texts}')
    print(f'From {total_verdicts} unique verdicts across {len(verdict_texts_by_dir)} directories')
    print(f'\nAggregating texts per verdict and sending to GPT...')
    print('='*60)

    # Process each directory separately
    for csv_dir, verdict_texts in verdict_texts_by_dir.items():
        # Use parent directory name for unique checkpoint (weapon/drugs/5k)
        parent_dir = os.path.basename(os.path.dirname(csv_dir))
        dir_name = f"{parent_dir}_{os.path.basename(csv_dir)}"
        print(f'\n📁 Processing directory: {dir_name}')
        
        # Checkpoint path per directory (now unique: weapon_verdict_csv, drugs_verdict_csv, 5k_verdict_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_path = os.path.join(output_dir, f'gpt_fewshot_classifier_extractor_checkpoint_{dir_name}.csv')
        else:
            checkpoint_path = f'gpt_fewshot_classifier_extractor_checkpoint_{dir_name}.csv'
        processed_verdicts = set()
        if os.path.exists(checkpoint_path):
            try:
                checkpoint_df = pd.read_csv(checkpoint_path)
                if not checkpoint_df.empty and 'verdict' in checkpoint_df.columns:
                    processed_verdicts = set(checkpoint_df['verdict'].astype(str))
                    print(f'📋 Resuming from checkpoint: {len(processed_verdicts)} verdicts already processed')
                # else: empty or malformed checkpoint, start fresh
            except Exception as e:
                # empty file or no columns - start fresh without warning
                if 'No columns to parse' not in str(e):
                    print(f'⚠️  Could not load checkpoint: {e}')
        
        results = []
        for verdict, texts in tqdm(verdict_texts.items(), desc=f"GPT processing {dir_name}"):
            if not texts:
                continue
            
            # Skip if already processed
            if verdict in processed_verdicts:
                continue
            
            # Deduplicate very similar texts (to avoid repetition)
            unique_texts = []
            seen_texts = set()
            for text in texts:
                # Normalize text for comparison (remove extra whitespace)
                normalized = ' '.join(text.split())
                # Skip if we've seen this exact text before
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    unique_texts.append(text)
            
            # Combine all unique texts with clear separator
            combined_text = '\n\n---\n\n'.join(unique_texts)
            
            try:
                gpt_result = gpt_classify_and_extract(combined_text)
                
                result = {
                    'verdict': verdict,
                    'text': combined_text,
                    'num_texts_combined': len(texts),
                    'classification': gpt_result['classification'],
                    'sentence': gpt_result['sentence'],
                    'confidence': gpt_result['confidence'],
                    'range_low': gpt_result['range'][0] if gpt_result['range'] else None,
                    'range_high': gpt_result['range'][1] if gpt_result['range'] else None,
                    'range_str': gpt_result['range_str'],
                    'gpt_output': gpt_result['gpt_output']
                }
                results.append(result)
                
                # Save checkpoint after each verdict (for large batches)
                if len(results) % 10 == 0:  # Save every 10 verdicts
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_df.to_csv(checkpoint_path, index=False)
            except Exception as e:
                print(f'\n⚠️  Error processing {verdict}: {e}')
                result = {
                    'verdict': verdict,
                    'text': combined_text[:100] + '...' if len(combined_text) > 100 else combined_text,
                    'num_texts_combined': len(texts),
                    'classification': 'ERROR',
                    'sentence': None,
                    'confidence': None,
                    'range_low': None,
                    'range_high': None,
                    'range_str': None,
                    'gpt_output': f'Error: {str(e)}'
                }
                results.append(result)
        
        # Save results for this directory (merge with checkpoint if exists)
        if output_dir:
            output_path = os.path.join(output_dir, f'gpt_fewshot_classifier_extractor_results_{dir_name}.csv')
        else:
            output_path = f'gpt_fewshot_classifier_extractor_results_{dir_name}.csv'
        if os.path.exists(checkpoint_path) and processed_verdicts:
            # Load existing results and merge
            try:
                existing_df = pd.read_csv(checkpoint_path)
                df_results = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
                # Remove duplicates (keep latest)
                df_results = df_results.drop_duplicates(subset=['verdict'], keep='last')
            except Exception as e:
                print(f'⚠️  Could not merge with checkpoint: {e}')
                df_results = pd.DataFrame(results)
        else:
            df_results = pd.DataFrame(results)
        
        df_results.to_csv(output_path, index=False)
        # Also update checkpoint
        df_results.to_csv(checkpoint_path, index=False)
        
        print(f'\n✅ Results saved to {output_path}')
        print(f'\nSummary for {dir_name}:')
        print(f'  Total processed: {len(df_results)}')
        if len(df_results) > 0 and 'classification' in df_results.columns:
            print(f'  Positive classifications: {len(df_results[df_results["classification"] == "POSITIVE"])}')
            print(f'  Negative classifications: {len(df_results[df_results["classification"] == "NEGATIVE"])}')
            if 'range_low' in df_results.columns:
                print(f'  With extracted ranges: {len(df_results[df_results["range_low"].notna()])}')
        else:
            print('  ⚠️  No results to summarize (empty DataFrame)')

    print('\n✅ Done! All directories processed.')


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract sentencing ranges from legal verdicts using GPT few-shot learning"
    )
    parser.add_argument(
        "--csv-dir", 
        type=str, 
        help="Path to CSV directory containing verdict files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Output directory for results (default: current directory)"
    )
    parser.add_argument(
        "--filter-by-target", 
        action="store_true",
        help="Only process verdicts in GT files"
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all verdicts (ignore GT filtering)"
    )
    parser.add_argument(
        "--target-list",
        type=str,
        default=None,
        help="Path to file with verdict IDs to process (one per line or CSV with 'verdict' column). Used with --filter-by-target."
    )
    
    args = parser.parse_args()
    
    # Determine CSV directories to process (resolve to absolute so path is correct regardless of cwd)
    if args.csv_dir:
        csv_dir_abs = os.path.abspath(os.path.expanduser(args.csv_dir))
        if not os.path.isdir(csv_dir_abs):
            print(f"❌ CSV directory not found: {csv_dir_abs}")
            print(f"   (resolved from: {args.csv_dir})")
            sys.exit(1)
        csv_dirs = [csv_dir_abs]
        print(f"📂 Using provided CSV directory: {csv_dir_abs}")
    else:
        csv_dirs = DEFAULT_CSV_DIRS
        print(f"📂 Using default CSV directories")
    
    # Load target verdicts: from --target-list file, or from GT files
    target_verdicts = set()
    if args.target_list and os.path.exists(args.target_list):
        try:
            with open(args.target_list, encoding="utf-8") as f:
                for line in f:
                    v = line.strip()
                    if v and not v.startswith("#"):
                        target_verdicts.add(v)
            if not target_verdicts:
                try:
                    df_t = pd.read_csv(args.target_list)
                    if "verdict" in df_t.columns:
                        target_verdicts = set(df_t["verdict"].dropna().astype(str).str.strip())
                except Exception:
                    pass
            print(f"✅ Loaded {len(target_verdicts)} verdicts from target list: {args.target_list}")
        except Exception as e:
            print(f"⚠️  Failed to load target list {args.target_list}: {e}")
    if not target_verdicts:
        for gt_file in GT_FILES:
            if os.path.exists(gt_file):
                try:
                    gt_df = pd.read_csv(gt_file)
                    if 'case' in gt_df.columns:
                        target_verdicts.update(gt_df['case'].dropna().astype(str).str.strip())
                    elif 'verdict' in gt_df.columns:
                        target_verdicts.update(gt_df['verdict'].dropna().astype(str).str.strip())
                    print(f"✅ Loaded {len(target_verdicts)} verdicts from GT file: {os.path.basename(gt_file)}")
                except Exception as e:
                    print(f"⚠️  Failed to load GT file {gt_file}: {e}")
    
    # Determine filtering mode
    filter_by_target = args.filter_by_target and not args.process_all
    
    if target_verdicts and filter_by_target:
        print(f"✅ Total {len(target_verdicts)} unique verdicts from target files (FILTERING ENABLED)")
    else:
        if target_verdicts:
            print(f"ℹ️  Found {len(target_verdicts)} verdicts in target files, but processing ALL verdicts")
        else:
            print(f"⚠️  No target verdicts found, processing all verdicts")
    
    # Run processing
    process_directories(
        csv_dirs=csv_dirs,
        output_dir=args.output_dir,
        filter_by_target=filter_by_target,
        target_verdicts=target_verdicts
    )


if __name__ == "__main__":
    main()


