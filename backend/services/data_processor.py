# This file will contain the data processing logic for uploaded files. 

import yaml
import pandas as pd
import re
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
from fastapi import UploadFile
import math
from datetime import datetime
from backend.services.time_series_processor import TimeSeriesProcessor
from sqlalchemy import select, func, cast, Integer

async def get_max_interim(db, table, monitor_id_field, monitor_id):
    # First, get all distinct interim values for this monitor
    distinct_query = (
        select(table.interim)
        .where(getattr(table, monitor_id_field) == monitor_id)
        .distinct()
    )
    result = await db.execute(distinct_query)
    interim_values = [row[0] for row in result.fetchall()]
    
    if not interim_values:
        return None
    
    # Extract numeric parts and find the maximum
    max_num = 0
    max_interim = None
    
    for interim in interim_values:
        # Extract numeric part from interim (e.g., "Interim1" -> 1, "Interim2" -> 2)
        import re
        match = re.search(r'\d+', interim)
        if match:
            num = int(match.group())
            if num > max_num:
                max_num = num
                max_interim = interim
    
    return max_interim

def ensure_actions_is_object(val, old_val=None):
    """
    Ensures the actions field is always a JSON object with the correct structure.
    If val is a string, wraps it as {"actions": val}.
    If val is a dict, returns as is.
    If val is None, returns None.
    Otherwise, coerces to string and wraps.
    
    Args:
        val: The new value to process
        old_val: The existing value (for update operations). If provided and val is a string,
                the string will be merged into the existing object as {"actions": val}
    """
    if val is None:
        return None
    if isinstance(val, str):
        if old_val and isinstance(old_val, dict):
            # For updates: merge the new string into existing object
            result = old_val.copy()
            result["actions"] = val
            return result
        else:
            # For creates: create new object
            return {"actions": val}
    if isinstance(val, dict):
        return val
    return {"actions": str(val)}

class DataProcessor:
    def __init__(self, config_path='backend/schemas/column_mappings.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.model_config = self.config.get('monitor', {}).get('fields', {})
        self.field_mappings = self._prepare_field_mappings()

    def _prepare_field_mappings(self) -> Dict[str, Any]:
        """
        Pre-processes the config for easier lookup.
        Creates a map from various raw patterns to a standard internal field name (the YAML key).
        """
        mappings = {}
        model_name = 'monitor'
        mappings[model_name] = {}
        for field_name, field_props in self.model_config.items():
            # The key in the YAML ('monitor_id', 'install_date', etc.) is the standard name.
            # The 'aliases' field was causing confusion and has been removed from the logic.
            for pattern in field_props.get('raw_patterns', []):
                try:
                    # More robust regex building
                    if not any(c in pattern for c in r'.*+?^${}()|[]\\'):
                        forgiving_pattern = r'\s*'.join(re.escape(p) for p in re.split(r'(\s+)', pattern) if p.strip())
                    else:
                        forgiving_pattern = pattern
                    mappings[model_name][re.compile(forgiving_pattern, re.IGNORECASE)] = field_name
                except re.error as e:
                    print(f"Skipping invalid regex pattern '{pattern}' for field '{field_name}': {e}")
        return mappings

    def _normalize_header(self, header: str) -> str:
        if not isinstance(header, str):
            return ""
        return re.sub(r'\s+', ' ', header).strip()

    def _find_best_sheet_and_header(self, file_content: BytesIO, file_type: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
        best_sheet_name = None
        best_header_row = -1
        max_score = -1
        best_column_map = {}

        xls = None
        if file_type == 'excel':
            try:
                xls = pd.ExcelFile(file_content)
                sheet_names = xls.sheet_names
            except Exception as e:
                print(f"Error reading Excel file: {e}")
                return None, None
        else: # csv
            sheet_names = ['csv_file']

        model_name = 'monitor'
        patterns_to_standard_name = self.field_mappings.get(model_name, {})
        
        for sheet_name in sheet_names:
            try:
                if file_type == 'excel':
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                else:
                    file_content.seek(0)
                    df_sheet = pd.read_csv(file_content, header=None)
            except Exception as e:
                print(f"Could not read sheet: {sheet_name}. Error: {e}")
                continue
            
            # Limit scan to top 15 rows
            for i, row in df_sheet.head(15).iterrows():
                current_score = 0
                current_column_map = {}
                headers = [str(h) for h in row.tolist()]

                for header in headers:
                    normalized_h = self._normalize_header(header)
                    if not normalized_h:
                        continue
                    
                    for pattern, standard_name in patterns_to_standard_name.items():
                        if pattern.fullmatch(normalized_h):
                            current_score += 1
                            if standard_name not in current_column_map:
                                current_column_map[standard_name] = header # Store raw header
                            break 

                if current_score > max_score:
                    max_score = current_score
                    best_sheet_name = sheet_name
                    best_header_row = i
                    best_column_map = current_column_map

        # Require a minimum score to proceed
        if max_score > 2: # At least 3 columns matched
            print(f"Best match found: sheet='{best_sheet_name}', header_row={best_header_row}, score={max_score}")
            file_content.seek(0)
            if file_type == 'excel':
                df_final = pd.read_excel(file_content, sheet_name=best_sheet_name, header=best_header_row, dtype=str)
            else:
                df_final = pd.read_csv(file_content, header=best_header_row, dtype=str)
            
            raw_to_standard_map = {v: k for k, v in best_column_map.items()}
            df_final.rename(columns=raw_to_standard_map, inplace=True)
            
            # Filter to only include columns present in our mapping
            valid_columns = [col for col in df_final.columns if col in self.model_config]
            return df_final[valid_columns], best_column_map
        
        return None, None

    def _clean_and_validate(self, value: Any, config: Dict[str, Any]) -> Any:
        """Applies cleaning and validation rules from the config."""
        if pd.isna(value) or value is None or str(value).strip() == '':
            return config.get('default')

        # Type conversion
        target_type = config.get('target_type')
        try:
            if target_type == 'integer':
                return int(float(value))
            elif target_type == 'float':
                return float(value)
            elif target_type == 'datetime':
                # More flexible datetime parsing, ignores the format from config
                return pd.to_datetime(value).to_pydatetime()
            else: # string or other
                return str(value).strip()
        except (ValueError, TypeError) as e:
            print(f"Type conversion error for value '{value}' to '{target_type}': {e}")
            return config.get('default')

    async def process_file(self, file: UploadFile, model_type: str, area: str) -> List[Dict[str, Any]]:
        # Save to minIO
        ts_processor = TimeSeriesProcessor()
        minio_file_url, minio_object_name = await ts_processor.save_file(file, area, model_type)
        # Re-read file content (because file has been read after save_file)
        file.file.seek(0)
        file_content = BytesIO(await file.read())
        filename = file.filename.lower()
        
        file_type = 'csv' if filename.endswith('.csv') else 'excel' if filename.endswith(('.xlsx', '.xls','.xlsm')) else None
        if file_type is None:
            raise ValueError("Unsupported file type")

        df, column_map = self._find_best_sheet_and_header(file_content, file_type)

        if df is None:
            raise ValueError("Could not find a suitable header in the file.")
            
        processed_data = []
        for index, row in df.iterrows():
            # Rule: Ignore rows where the primary identifier (monitor_id) is missing.
            # This check is now more robust to handle various 'empty' values from Excel.
            monitor_id_val = None
            if 'monitor_id' in df.columns:
                raw_monitor_id = row.get('monitor_id')
                if pd.notna(raw_monitor_id):
                    # Ensure it's not just whitespace or 'nan'/'none' strings
                    id_str = str(raw_monitor_id).strip()
                    if id_str and id_str.lower() not in ['nan', 'none']:
                        monitor_id_val = id_str

            if not monitor_id_val:
                continue # Skip rows with any form of empty monitor_id
            
            # Rule: monitor_id must be a combination of letters and numbers (e.g. PL01, FM25)
            if not re.match(r'^[a-zA-Z]+[0-9]+$', str(monitor_id_val)):
                continue

            # Rule 1: Special handling for 'scrapped' devices.
            is_scrapped = any(str(cell).strip().lower() == 'scrapped' for cell in row.values)

            if is_scrapped:
                scrapped_record = {
                    'monitor_id': monitor_id_val,
                    'monitor_name': monitor_id_val,
                    'type': model_type,
                    'area': area,
                    'status': 'scrapped'
                }
                processed_data.append(scrapped_record)
                continue

            # Default processing for a valid, non-scrapped row.
            record = {
                'type': model_type,
                'area': area,
                'monitor_id': monitor_id_val,
                'monitor_name': monitor_id_val,
                'status': 'uninstalled'
            }
            
            is_row_valid = True
            for standard_name, field_conf in self.model_config.items():
                # Skip identifiers as they are already handled
                if standard_name in ['monitor_id', 'monitor_name']:
                    continue

                if standard_name in df.columns:
                    raw_value = row.get(standard_name)
                    cleaned_value = self._clean_and_validate(raw_value, field_conf)
                    
                    if field_conf.get('required', False) and cleaned_value is None:
                        # For required fields, if cleaning results in None, invalidate the row
                        print(f"Row {index+2}: Missing or invalid required value for '{standard_name}'")
                        is_row_valid = False
                        break 
                    
                    record[standard_name] = cleaned_value
            
            if is_row_valid:
                processed_data.append(record)
            
        # Only return processed data, not minIO information
        return processed_data 