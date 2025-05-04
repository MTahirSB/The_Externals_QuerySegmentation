import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import re
import multiprocessing as mp
from functools import partial, lru_cache
import os
import json
import logging
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time
from pathlib import Path
import hashlib

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bioes_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Global caches with size limits to prevent memory issues
TEXT_CLEANING_CACHE = {}
TOKENIZER_CACHE = {}
ATTRIBUTE_PARSING_CACHE = {}
MAX_CACHE_SIZE = 100000  # Adjust based on your memory constraints

# Performance metrics
METRICS = {
    'start_time': None,
    'rows_processed': 0,
    'chunks_processed': 0,
    'errors': 0
}

class CacheManager:
    """Manages caches with size limits to prevent memory overflow"""
    
    @staticmethod
    def trim_cache(cache: Dict, max_size: int) -> None:
        """Trim cache to maximum size if needed"""
        if len(cache) > max_size:
            # Remove approximately 20% of the cache (oldest items in a FIFO manner)
            items_to_remove = int(max_size * 0.2)
            keys_to_remove = list(cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del cache[key]
            logger.debug(f"Trimmed cache, removed {len(keys_to_remove)} entries")

    @staticmethod
    def clear_all_caches() -> None:
        """Clear all caches"""
        TEXT_CLEANING_CACHE.clear()
        TOKENIZER_CACHE.clear()
        ATTRIBUTE_PARSING_CACHE.clear()
        logger.debug("All caches cleared")

@lru_cache(maxsize=10000)
def clean_text(text: str) -> str:
    """Clean text with caching and improved cleaning"""
    if pd.isna(text) or text == '':
        return ""
    
    text_str = str(text)
    if text_str in TEXT_CLEANING_CACHE:
        return TEXT_CLEANING_CACHE[text_str]
    
    # Enhanced text cleaning
    cleaned = str(text).strip()
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Handle punctuation with proper spacing
    cleaned = re.sub(r'([.,!?;:])\s*', r'\1 ', cleaned)
    # Normalize quotes for consistency
    cleaned = re.sub(r'[""`´'']', "'", cleaned)
    # Final whitespace cleanup
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    TEXT_CLEANING_CACHE[text_str] = cleaned
    CacheManager.trim_cache(TEXT_CLEANING_CACHE, MAX_CACHE_SIZE)
    return cleaned

def parse_attributes(attr_str: str) -> List[str]:
    """Parse attributes with improved handling for various formats"""
    if pd.isna(attr_str) or attr_str in ('', '[]', '{}', '()', 'nan'):
        return []
    
    # Check cache
    if attr_str in ATTRIBUTE_PARSING_CACHE:
        return ATTRIBUTE_PARSING_CACHE[attr_str]
    
    # Try different parsing strategies
    result = []
    
    # Strategy 1: Try direct JSON parsing
    try:
        parsed = json.loads(attr_str)
        if isinstance(parsed, list):
            result = [clean_text(item) for item in parsed if item and not pd.isna(item)]
        elif isinstance(parsed, dict):
            # Handle dict format by extracting values
            result = [clean_text(val) for val in parsed.values() if val and not pd.isna(val)]
    except (json.JSONDecodeError, TypeError):
        # Strategy 2: Try after normalizing quotes
        try:
            cleaned_str = attr_str.replace("'", '"')
            parsed = json.loads(cleaned_str)
            if isinstance(parsed, list):
                result = [clean_text(item) for item in parsed if item and not pd.isna(item)]
            elif isinstance(parsed, dict):
                result = [clean_text(val) for val in parsed.values() if val and not pd.isna(val)]
        except (json.JSONDecodeError, TypeError):
            # Strategy 3: Use regex patterns
            patterns = [
                r'[\'"]([^\'"]*)[\'"]',  # Match quoted strings
                r'[\[\{\(]([^\]\}\)]+)[\]\}\)]',  # Match content within brackets
                r'([^,\[\]\{\}\(\)]+)'  # Match any content between separators
            ]
            
            for pattern in patterns:
                attr_items = re.findall(pattern, attr_str)
                if attr_items:
                    result = [clean_text(item) for item in attr_items if item.strip() and not pd.isna(item)]
                    break
    
    # Remove duplicates while preserving order
    unique_results = []
    seen = set()
    for item in result:
        if item and item not in seen:
            seen.add(item)
            unique_results.append(item)
    
    # Cache and return results
    ATTRIBUTE_PARSING_CACHE[attr_str] = unique_results
    CacheManager.trim_cache(ATTRIBUTE_PARSING_CACHE, MAX_CACHE_SIZE)
    return unique_results

@lru_cache(maxsize=10000)
def tokenize_query(query: str) -> List[str]:
    """Enhanced tokenizer with better handling of punctuation and special characters"""
    if pd.isna(query) or query == '':
        return []
    
    # Check cache first
    clean_query = clean_text(query)
    if clean_query in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[clean_query]
    
    # Improved tokenization that preserves important punctuation
    # First, add spaces around punctuation we want to separate
    spaced_query = re.sub(r'([.,!?;:])', r' \1 ', clean_query)
    # Then, split on whitespace
    tokens = [token for token in spaced_query.split() if token]
    
    # Cache result
    TOKENIZER_CACHE[clean_query] = tokens
    CacheManager.trim_cache(TOKENIZER_CACHE, MAX_CACHE_SIZE)
    return tokens

def is_brand_in_query(brand: str, query: str) -> bool:
    """Check if a brand appears in the query (case insensitive)"""
    if not brand or not query:
        return False
    
    # Clean both strings for comparison
    clean_brand = clean_text(brand).lower()
    clean_query = clean_text(query).lower()
    
    # Check if brand name is in query
    return clean_brand in clean_query

def apply_bioes_tags(tokens: List[str], tags: List[str], entity: str, label: str) -> List[str]:
    """Apply BIOES tags with improved entity detection including partial matches"""
    if not entity:
        return tags
    
    entity_tokens = tokenize_query(entity)
    if not entity_tokens:
        return tags
    
    entity_lowercase = [t.lower() for t in entity_tokens]
    tokens_lowercase = [t.lower() for t in tokens]
    
    # Strategy 1: Try exact matches first
    for i in range(len(tokens) - len(entity_tokens) + 1):
        match = True
        for j in range(len(entity_tokens)):
            if i + j >= len(tokens) or tokens_lowercase[i + j] != entity_lowercase[j]:
                match = False
                break
                
        if match and all(tags[pos] == 'O' for pos in range(i, i + len(entity_tokens))):
            start, end = i, i + len(entity_tokens) - 1
            
            # Apply tags
            if start == end:  # Single token
                tags[start] = f'S-{label}'
            else:  # Multiple tokens
                tags[start] = f'B-{label}'
                for pos in range(start + 1, end):
                    tags[pos] = f'I-{label}'
                tags[end] = f'E-{label}'
            
            return tags
    
    # Strategy 2: Try fuzzy matching if exact match fails
    # This allows for entities with slight variations to be tagged
    if len(entity_tokens) > 2:  # Only for longer entities
        for i in range(len(tokens) - len(entity_tokens) + 1):
            # Calculate how many tokens match
            matches = sum(1 for j in range(len(entity_tokens)) 
                        if i + j < len(tokens) and tokens_lowercase[i + j] == entity_lowercase[j])
            
            # If at least 80% of tokens match and no existing tags
            match_ratio = matches / len(entity_tokens)
            if match_ratio >= 0.8 and all(tags[pos] == 'O' for pos in range(i, i + len(entity_tokens))):
                start, end = i, i + len(entity_tokens) - 1
                
                # Apply tags
                if start == end:
                    tags[start] = f'S-{label}'
                else:
                    tags[start] = f'B-{label}'
                    for pos in range(start + 1, end):
                        tags[pos] = f'I-{label}'
                    tags[end] = f'E-{label}'
                
                return tags
    
    return tags

def process_row(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single row with enhanced entity tagging"""
    try:
        # Extract and normalize column names
        query = row_data.get('query', row_data.get('Query', ''))
        brand = row_data.get('brand', row_data.get('Brand', ''))
        product_type = row_data.get('product_type', row_data.get('Product_type', ''))
        
        # Handle attributes with improved parsing
        attributes_raw = row_data.get('attributes', row_data.get('Attributes', []))
        if isinstance(attributes_raw, str):
            attributes = parse_attributes(attributes_raw)
        elif isinstance(attributes_raw, list):
            attributes = [clean_text(attr) for attr in attributes_raw if attr and not pd.isna(attr)]
        else:
            attributes = []
        
        # Clean and tokenize the query
        original_query = clean_text(query)
        tokens = tokenize_query(original_query)
        
        # Check if brand is present in the query
        brand_in_query = is_brand_in_query(brand, original_query)
        
        # If the brand is not in the query but present in the brand column, 
        # we need to add it to the tokens for tagging
        clean_brand = clean_text(brand)
        
        # Prepare result dictionary with original data
        result = {
            'query': original_query,
            'brand': clean_brand,
            'attributes': attributes,
            'product_type': clean_text(product_type)
        }
        
        # Skip processing if no tokens
        if not tokens:
            result.update({
                'tokens': [],
                'bioes_tags': [],
                'augmented_query': original_query,
                'brand_added': False
            })
            return result
        
        # Initialize tags as 'O'
        bioes_tags = ['O'] * len(tokens)
        
        # Apply tags in order of priority for the original query
        if brand and brand_in_query:
            bioes_tags = apply_bioes_tags(tokens, bioes_tags, clean_brand, 'Brand')
        
        # Apply attributes
        for attr in attributes:
            if attr:
                bioes_tags = apply_bioes_tags(tokens, bioes_tags, clean_text(attr), 'Attribute')
        
        # Apply product type
        if product_type:
            bioes_tags = apply_bioes_tags(tokens, bioes_tags, clean_text(product_type), 'ProductType')
        
        # If brand is not in query but exists, we handle it separately
        brand_added = False
        augmented_query = original_query
        augmented_tokens = tokens.copy()
        augmented_tags = bioes_tags.copy()
        
        if clean_brand and not brand_in_query:
            # Add the brand to the query for tagging purposes
            brand_added = True
            brand_tokens = tokenize_query(clean_brand)
            
            if brand_tokens:
                # Add the brand tokens at the end
                augmented_query = f"{original_query} {clean_brand}"
                augmented_tokens = tokens + brand_tokens
                
                # Extend tags list with 'O' for the brand tokens
                augmented_tags = bioes_tags + ['O'] * len(brand_tokens)
                
                # Now apply brand tags to the extended tags list
                start_idx = len(tokens)  # Start index for the brand tokens
                
                if len(brand_tokens) == 1:
                    augmented_tags[start_idx] = f'S-Brand'
                else:
                    augmented_tags[start_idx] = f'B-Brand'
                    for i in range(start_idx + 1, start_idx + len(brand_tokens) - 1):
                        augmented_tags[i] = f'I-Brand'
                    augmented_tags[start_idx + len(brand_tokens) - 1] = f'E-Brand'
        
        # Track metrics
        METRICS['rows_processed'] += 1
        
        # Only include necessary columns in the result
        # Removed 'original_tokens' and 'original_tags' as requested
        result.update({
            'tokens': augmented_tokens,
            'bioes_tags': augmented_tags,
            'augmented_query': augmented_query,
            'brand_added': brand_added
        })
        
        return result
    except Exception as e:
        METRICS['errors'] += 1
        logger.error(f"Error processing row {row_data.get('query', '')}: {e}")
        return {
            'query': row_data.get('query', row_data.get('Query', '')),
            'tokens': [],
            'bioes_tags': [],
            'error': str(e)
        }

def process_chunk(chunk: pd.DataFrame, process_pool=None) -> List[Dict[str, Any]]:
    """Process chunk with optional parallel processing"""
    if process_pool:
        # Prepare rows for parallel processing
        rows = [row.to_dict() for _, row in chunk.iterrows()]
        
        # Normalize column names
        for row_dict in rows:
            for old_key, new_key in [('Query', 'query'), ('Brand', 'brand'), 
                                    ('Product_type', 'product_type'), ('Attributes', 'attributes')]:
                if old_key in row_dict and new_key not in row_dict:
                    row_dict[new_key] = row_dict[old_key]
        
        # Process in parallel
        results = process_pool.map(process_row, rows)
    else:
        # Sequential processing
        results = []
        for _, row in chunk.iterrows():
            row_dict = row.to_dict()
            
            # Normalize column names
            for old_key, new_key in [('Query', 'query'), ('Brand', 'brand'), 
                                    ('Product_type', 'product_type'), ('Attributes', 'attributes')]:
                if old_key in row_dict and new_key not in row_dict:
                    row_dict[new_key] = row_dict[old_key]
            
            results.append(process_row(row_dict))
    
    METRICS['chunks_processed'] += 1
    return results

def read_data_chunked(input_file: str, chunksize: int = 100000):
    """Read data with enhanced error handling and format detection"""
    try:
        # Convert to Path object for better path handling
        input_path = Path(input_file)
        
        # Check if file exists
        if not input_path.exists():
            logger.error(f"File not found: {input_path}")
            raise FileNotFoundError(f"Input file '{input_path}' does not exist")
            
        # Check if file is empty
        if input_path.stat().st_size == 0:
            logger.error(f"File is empty: {input_path}")
            raise ValueError(f"Input file '{input_path}' is empty")
            
        logger.info(f"Starting to read file: {input_path}")
        
        # Process based on file extension
        if input_path.suffix.lower() == '.csv':
            try:
                # Try with default engine first
                for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
                    yield chunk
            except pd.errors.ParserError as e:
                logger.warning(f"Parser error with default engine: {e}. Trying python engine...")
                # Try with python engine and more robust settings
                for chunk in pd.read_csv(
                    input_path,
                    escapechar='\\',
                    engine='python',
                    on_bad_lines='warn',
                    chunksize=chunksize,
                    low_memory=False
                ):
                    yield chunk
            except UnicodeDecodeError:
                logger.warning("Unicode decode error, trying different encodings...")
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                    try:
                        logger.info(f"Trying encoding: {encoding}")
                        for chunk in pd.read_csv(
                            input_path, 
                            encoding=encoding,
                            chunksize=chunksize,
                            low_memory=False
                        ):
                            yield chunk
                        break  # Stop if successful
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error with encoding {encoding}: {e}")
                        continue
        
        elif input_path.suffix.lower() == '.json':
            try:
                # Try as line-delimited JSON first
                for chunk in pd.read_json(input_path, chunksize=chunksize, lines=True):
                    yield chunk
            except ValueError:
                # Try as regular JSON
                logger.info("Trying to read as regular JSON file")
                df = pd.read_json(input_path)
                # Yield chunks manually
                for i in range(0, len(df), chunksize):
                    yield df.iloc[i:i+chunksize]
            except UnicodeDecodeError:
                logger.warning("Unicode decode error in JSON, trying different encodings...")
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']:
                    try:
                        with open(input_path, 'r', encoding=encoding) as f:
                            data = json.load(f)
                        df = pd.DataFrame(data)
                        for i in range(0, len(df), chunksize):
                            yield df.iloc[i:i+chunksize]
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error with encoding {encoding}: {e}")
                        continue
        
        elif input_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                # For Excel files, try both openpyxl and xlrd engines
                try:
                    logger.info("Reading Excel file with openpyxl engine")
                    df = pd.read_excel(input_path, engine='openpyxl')
                except Exception as e:
                    logger.warning(f"Failed with openpyxl: {e}, trying xlrd engine")
                    df = pd.read_excel(input_path, engine='xlrd')
                
                # Yield chunks manually
                for i in range(0, len(df), chunksize):
                    yield df.iloc[i:i+chunksize]
            except Exception as e:
                logger.error(f"Failed to read Excel file: {e}")
                raise
        
        else:
            supported_formats = "CSV (.csv), JSON (.json), or Excel (.xlsx, .xls)"
            raise ValueError(f"Unsupported file format for {input_path}. Use {supported_formats}.")
    
    except Exception as e:
        logger.error(f"Failed to read {input_file}: {str(e)}")
        raise

def validate_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize column names"""
    required_cols = ['query', 'brand', 'product_type', 'attributes']
    alt_cols = ['Query', 'Brand', 'Product_type', 'Attributes']
    
    # Check which columns are present and normalize
    for req_col, alt_col in zip(required_cols, alt_cols):
        if req_col not in df.columns and alt_col in df.columns:
            df[req_col] = df[alt_col]
        elif req_col not in df.columns and alt_col not in df.columns:
            logger.warning(f"Column {req_col}/{alt_col} not found in data. Adding empty column.")
            df[req_col] = ""
    
    return df

def generate_bioes_sequences(
    input_file: str, 
    output_file: str, 
    chunk_size: int = 10000,
    use_parallel: bool = True,
    num_workers: int = None
) -> None:
    """Main processing function with enhanced features - stats generation removed"""
    # Start timing
    METRICS['start_time'] = time.time()
    
    logger.info(f"Reading data from {input_file} in chunks of {chunk_size}")
    logger.info(f"Parallel processing: {use_parallel}")
    
    # Determine reasonable number of workers if not specified
    if use_parallel and num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        logger.info(f"Using {num_workers} workers for parallel processing")
    
    # Create output directory if needed
    output_path = Path(output_file)
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize process pool if using parallel processing
    process_pool = None
    if use_parallel:
        process_pool = mp.Pool(processes=num_workers)
    
    first_chunk = True
    total_rows = 0
    
    try:
        # Process in chunks
        chunk_iterator = read_data_chunked(input_file, chunk_size)
        
        # Use tqdm to display progress
        pbar = tqdm(desc="Processing chunks", unit="chunk")
        
        for chunk in chunk_iterator:
            pbar.update(1)
            logger.info(f"Processing chunk with {len(chunk)} rows")
            
            # Validate and normalize column names
            chunk = validate_column_names(chunk)
            
            # Process the chunk
            chunk_results = process_chunk(chunk, process_pool if use_parallel else None)
            total_rows += len(chunk_results)
            
            # Convert to DataFrame
            chunk_df = pd.DataFrame(chunk_results)
            
            # Write to output file
            if output_file.lower().endswith('.csv'):
                if first_chunk:
                    chunk_df.to_csv(output_file, index=False)
                    first_chunk = False
                else:
                    chunk_df.to_csv(output_file, mode='a', header=False, index=False)
            elif output_file.lower().endswith('.json'):
                if first_chunk:
                    chunk_df.to_json(output_file, orient='records', lines=True)
                    first_chunk = False
                else:
                    # Use line-by-line JSON for better append performance
                    chunk_df.to_json(output_file, orient='records', lines=True, mode='a')
            else:
                # For Excel we collect all results and write at the end
                if first_chunk:
                    full_results_df = chunk_df
                    first_chunk = False
                else:
                    full_results_df = pd.concat([full_results_df, chunk_df], ignore_index=True)
            
            # Periodically clean caches to prevent memory issues
            if METRICS['chunks_processed'] % 10 == 0:
                CacheManager.trim_cache(TEXT_CLEANING_CACHE, MAX_CACHE_SIZE)
                CacheManager.trim_cache(TOKENIZER_CACHE, MAX_CACHE_SIZE)
                CacheManager.trim_cache(ATTRIBUTE_PARSING_CACHE, MAX_CACHE_SIZE)
        
        pbar.close()
        
        # For Excel, write the file at the end
        if output_file.lower().endswith(('.xlsx', '.xls')):
            if 'full_results_df' in locals():
                full_results_df.to_excel(output_file, index=False, engine='openpyxl')
        
        processing_time = time.time() - METRICS['start_time']
        logger.info(f"Completed processing {total_rows} rows in {processing_time:.2f} seconds")
        logger.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed during processing: {e}")
        raise
    
    finally:
        # Close process pool if it was created
        if process_pool:
            process_pool.close()
            process_pool.join()
        
        # Clear caches
        CacheManager.clear_all_caches()

def main():
    """Main function with improved command line interface"""
    import argparse
    
    # Reset metrics
    METRICS['start_time'] = None
    METRICS['rows_processed'] = 0
    METRICS['chunks_processed'] = 0
    METRICS['errors'] = 0
    
    # Clear caches to start with fresh memory
    CacheManager.clear_all_caches()
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate BIOES sequences from query data')
    parser.add_argument('--input', '-i', type=str, required=False, 
                        default='/Users/mtsb/Documents/QuerySequencing1/data/Brand_Inference.csv',
                        help='Input file path (CSV, JSON, or Excel)')
    parser.add_argument('--output', '-o', type=str, required=False, 
                        default="data/bioes_sequences.csv",
                        help='Output file path (CSV, JSON, or Excel)')
    parser.add_argument('--chunk-size', '-c', type=int, default=10000,
                        help='Chunk size for processing large files')
    parser.add_argument('--parallel', '-p', action='store_true',
                        help='Use parallel processing')
    parser.add_argument('--workers', '-w', type=int, default=None,
                        help='Number of worker processes for parallel processing')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--cache-size', '-s', type=int, default=100000,
                        help='Maximum size for each cache')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Set cache size
    global MAX_CACHE_SIZE
    MAX_CACHE_SIZE = args.cache_size
    
    try:
        # Print startup message
        print(f"Starting BIOES sequence generation...")
        print(f"Input file: {args.input}")
        print(f"Output file: {args.output}")
        print(f"Chunk size: {args.chunk_size}")
        print(f"Using parallel processing: {args.parallel}")
        if args.parallel:
            print(f"Number of workers: {args.workers if args.workers else 'auto'}")
        
        # Run the main processing function
        generate_bioes_sequences(
            input_file=args.input,
            output_file=args.output,
            chunk_size=args.chunk_size,
            use_parallel=args.parallel,
            num_workers=args.workers
        )
        
        processing_time = time.time() - METRICS['start_time']
        print(f"\nProcessing complete. Results saved to {args.output}")
        print(f"Processed {METRICS['rows_processed']} rows in {processing_time:.2f} seconds")
        print(f"Processing speed: {METRICS['rows_processed']/processing_time:.2f} rows/second")
        print(f"Errors: {METRICS['errors']}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
def extract_entities(tokens, tags):
    """
    Given a list of tokens and their BIOES tags, returns a dict of extracted
    entities by type (Brand, Attribute, ProductType).
    """
    entities = {'Brand': [], 'Attribute': [], 'ProductType': []}
    curr, curr_type = [], None
    for token, tag in zip(tokens, tags):
        if tag.startswith('S-'):
            typ = tag.split('-', 1)[1]
            entities[typ].append(token)
        elif tag.startswith('B-'):
            curr_type = tag.split('-', 1)[1]
            curr = [token]
        elif tag.startswith('I-') and curr_type:
            curr.append(token)
        elif tag.startswith('E-') and curr_type:
            curr.append(token)
            entities[curr_type].append(" ".join(curr))
            curr_type, curr = None, []
        else:
            curr_type, curr = None, []
    return entities


if __name__ == "__main__":
    main()
