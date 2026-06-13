# analysis_representations — ניתוח עומק של הפיצ'רים שחולצו אוטומטית

ניתוח מבוסס-נתונים של ארבעת ייצוגי-הפיצ'רים האוטומטיים (Hybrid-Manual, Hybrid-Full,
GPT-Law, GPT-Free) על שני הדומיינים (סמים, נשק). הפרק המלא: `CHAPTER_feature_deep_analysis.md`.

## הרצה (reproducibility)
דורש `OPENAI_API_KEY` ב-`experiments/.env`. הריצו לפי הסדר:

```bash
python3 inventory.py            # טבלה 1: שמות-שדה, יציבות, singletons -> out/inventory_summary.csv
python3 tune_threshold.py       # בחירת סף ה-clustering (אבחון בלבד)
python3 embed_cluster.py        # embeddings + clustering -> out/concepts_{domain}.csv, out/key2cluster_{domain}.json
python3 classify_beyond_schema.py  # מיפוי SCHEMA/BEYOND/IDENT -> out/beyond_schema_report.txt, out/schema_map_{domain}.json
python3 hybrid_depth.py         # אנטומיית ההעשרה -> out/hybrid_depth_summary.csv
python3 enrichment_split.py     # נטרול הליבה הידנית המוזרקת (פילוח מסת-מידע של תוספות-GPT)
```

## קבצים
| קובץ | תפקיד |
|---|---|
| `common.py` | טעינת וקטורי-פיצ'רים פר-תיק (dedup של זוגות מ-`similarity_database_*.csv`) |
| `inventory.py` | ספירת שמות-שדה, יציבות, singletons + נירמול שמות (`normalize`) |
| `tune_threshold.py` | sweep לבחירת סף ה-clustering (0.18) |
| `embed_cluster.py` | איחוד סמנטי (OpenAI embeddings + agglomerative), מושגים, חפיפה |
| `classify_beyond_schema.py` | מיפוי כל מושג ל-SCHEMA/BEYOND/IDENT (LLM) |
| `hybrid_depth.py` | ליבה מול העשרה, חד-פעמיים, ערכים טריוויאליים |
| `enrichment_split.py` | נטרול הליבה הידנית — פילוח מסת-מידע של תוספות-GPT בלבד |

## הערות מתודולוגיה
- `normalize()` מקלף **רק** את "האם" (מילת-שאלה). אינו מקלף מילות-תוכן כמו מספר/סוג/כמות/מקום
  כי הן מבחינות בין שדות-סכמה שונים (למשל "מספר עבירה" מול "סוג עבירה"). איחוד נרדפים אמיתיים
  נעשה ע"י ה-clustering הסמנטי, לא ע"י קילוף.
- `out/emb_cache.json` (~144MB) אינו ב-git; נוצר-מחדש ע"י `embed_cluster.py`.
