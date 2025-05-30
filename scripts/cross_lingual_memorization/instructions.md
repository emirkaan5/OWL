There are five steps to creating cross-lingual data.
1. Create CLM columns using MSFT Translator
    script: translator.py
    goal: create columns (st, yo, mai, mg, tn, ty)

2. Check CLM columns for english sentences / hallucinations using langdetect & polyglot
    script: catch_english.py
    goal: create columns (st_eng_detect, yo_eng_detect, mai_eng_detect, mg_eng_detect, tn_eng_detect, ty_eng_detect)

3. Check CLM columns for repettitons
    script: catch_repeats.py
    goal: create columns (st_has_repeated_ngram, yo_has_repeated_ngram, mai_has_repeated_ngram, mg_has_repeated_ngram, tn_has_repeated_ngram, ty_has_repeated_ngram)

4. Check CLM Masked columns for @@delimited@@ n-gram counts
    script: catch_delimitors.py
    goal: create columns (st_placeholder_count, yo_placeholder_count, mai_placeholder_count, mg_placeholder_count, tn_placeholder_count, ty_placeholder_count)

5. For any columns where {lang}_placeholder_count > en_placeholder_count, {lang}_has_repeated_ngram = True, or {lang}_eng_detect = True, replace translation with google translate
    script: google_translate.py
    goal: create columns (st_gt, yo_gt, mai_gt, mg_gt, tn_gt, ty_gt)

For any columns replaced with gt, add row # and file name to markers.csv.