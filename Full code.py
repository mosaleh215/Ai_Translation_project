# ! pip install deep_translator translatepy googletrans
import time
import pandas as pd
import numpy as np
import os
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")

INPUT_CSV = "/kaggle/input/weeeee/sbs_lookups_hotels.csv"
OUTPUT_CSV = "/kaggle/working/hotels_translated.csv"
CACHE_FILE = "/kaggle/working/translation_cache.json"
LIMIT_ROWS = None  #for all datarows, change it as you want
CHUNK_SIZE = 5000  
MAX_WORKERS = 20  
BATCH_SIZE = 10  


translation_cache = {}
cache_lock = threading.Lock()

def load_cache():
    """Load translation cache from file if it exists"""
    global translation_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                translation_cache = json.load(f)
            print(f"Loaded {len(translation_cache)} cached translations")
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            translation_cache = {}

def save_cache():
    """Save translation cache to file"""
    with cache_lock:
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(translation_cache, f, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

def get_cache_key(text, target_lang):
    """Generate a unique key for caching translations"""
    return hashlib.md5(f"{text}_{target_lang}".encode()).hexdigest()

def translate_text_fast(text, target_lang="ar"):
    """Fast translation using multiple services with fallbacks"""
    if not text or not text.strip():
        return ""
    
    cache_key = get_cache_key(text, target_lang)
    
    # Check cache first
    with cache_lock:
        if cache_key in translation_cache:
            return translation_cache[cache_key]
    
    result = ""
    
    try:
        from deep_translator import GoogleTranslator, BingTranslator, YandexTranslator
        
        try:
            translator = GoogleTranslator(source="en", target=target_lang)
            result = translator.translate(text)
        except:
            try:
                translator = BingTranslator(source="en", target=target_lang)
                result = translator.translate(text)
            except:
                try:
                    translator = YandexTranslator(source="en", target=target_lang)
                    result = translator.translate(text)
                except:
                    pass
    except ImportError:
        pass
    
    if not result:
        try:
            from translatepy import Translator
            translator = Translator()
            result = translator.translate(text, target_lang).result
        except:
            pass
    
    if not result:
        try:
            from googletrans import Translator
            translator = Translator()
            result = translator.translate(text, dest=target_lang).text
        except:
            pass
    
    with cache_lock:
        translation_cache[cache_key] = result
    
    return result

def translate_batch_fast(texts, target_lang="ar"):
    """Translate a batch of texts in parallel"""
    if not texts or not any(texts):
        return ["" for _ in texts]
    
    non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    non_empty_texts = [texts[i] for i in non_empty_indices]
    
    if not non_empty_texts:
        return ["" for _ in texts]
    
    results = ["" for _ in texts]
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(translate_text_fast, text, target_lang): i 
            for i, text in zip(non_empty_indices, non_empty_texts)
        }
        
        for future in as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                result = future.result()
                results[original_index] = result
            except Exception as e:
                print(f"Translation error: {str(e)}")
                results[original_index] = ""
    
    return results

def parallel_translate_fast(series, target_lang="ar", desc="Translating"):
    """Translate a pandas Series in parallel with progress bar"""
    batches = [series[i:i+BATCH_SIZE].tolist() 
               for i in range(0, len(series), BATCH_SIZE)]
    
    results = []
    with tqdm(total=len(batches), desc=desc) as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_batch = {
                executor.submit(translate_batch_fast, batch, target_lang): i 
                for i, batch in enumerate(batches)
            }
            
            batch_results = [None] * len(batches)
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    batch_results[batch_index] = future.result()
                except Exception as e:
                    print(f"Batch translation error: {str(e)}")
                    batch_results[batch_index] = [""] * len(batches[batch_index])
                pbar.update(1)
            
            for batch_result in batch_results:
                results.extend(batch_result)
    
    return pd.Series(results, index=series.index)

def process_chunk(chunk, chunk_num):
    """Process a single chunk of data"""
    print(f"\nProcessing chunk {chunk_num} with {len(chunk)} rows")
    
    chunk["name_ar"] = parallel_translate_fast(chunk["name"], desc=f"Translating names (chunk {chunk_num})")
    chunk["description_ar"] = parallel_translate_fast(chunk["description"], desc=f"Translating descriptions (chunk {chunk_num})")
    
    desired_order = [
        "name", "name_ar", "description", "description_ar",
        "id", "provider_hotel_code", "provider_id", "country_id",
        "city_id", "lat", "long", "address", "postal_code",
        "created_at", "updated_at"
    ]
    
    final_cols = [c for c in desired_order if c in chunk.columns]
    leftover = [c for c in chunk.columns if c not in final_cols]
    chunk = chunk[final_cols + leftover]
    
    return chunk

def main():
    start_time = time.time()
    
    load_cache()
    
    output_exists = os.path.exists(OUTPUT_CSV)
    processed_rows = 0
    
    if output_exists:
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            processed_rows = sum(1 for _ in f) - 1  
        print(f"Resuming from row {processed_rows}")
    
    reader = pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)
    first_chunk = True
    chunk_num = 0
    
    for chunk in reader:
        chunk_num += 1
        
        if processed_rows > 0:
            if len(chunk) <= processed_rows:
                processed_rows -= len(chunk)
                continue
            else:
                chunk = chunk.iloc[processed_rows:]
                processed_rows = 0
        
        if LIMIT_ROWS and chunk_num * CHUNK_SIZE > LIMIT_ROWS:
            remaining = LIMIT_ROWS - (chunk_num - 1) * CHUNK_SIZE
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        
        processed_chunk = process_chunk(chunk, chunk_num)
        
        if first_chunk or not output_exists:
            processed_chunk.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            first_chunk = False
            output_exists = True
        else:
            processed_chunk.to_csv(OUTPUT_CSV, mode='a', header=False, index=False, encoding="utf-8-sig")
        
        save_cache()
        
        print(f"Saved chunk {chunk_num}. Total processed: {len(processed_chunk) * chunk_num} rows")
    
    save_cache()
    
    elapsed = time.time() - start_time
    print(f"\nDone! Total time: {elapsed/60:.2f} minutes")

if __name__ == "__main__":
    main()