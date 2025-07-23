"""
Log management script for machine learning module
Provides utilities to view, analyze, and manage log files
"""
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import gzip
import shutil
import json
from typing import List, Dict, Any
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.machine_learning.utils.logger_config import get_ml_logger

logger = get_ml_logger("log_manager")

LOGS_DIR = Path(__file__).parent / "logs"

class LogManager:
    """
    Manager for ML module log files
    """
    
    def __init__(self):
        self.logs_dir = LOGS_DIR
        self.logs_dir.mkdir(exist_ok=True)
    
    def list_log_files(self) -> List[Dict[str, Any]]:
        """List all log files with metadata"""
        log_files = []
        
        for log_file in self.logs_dir.glob("*.log*"):
            if log_file.is_file():
                stat = log_file.stat()
                log_info = {
                    'name': log_file.name,
                    'path': str(log_file),
                    'size_bytes': stat.st_size,
                    'size_mb': stat.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stat.st_mtime),
                    'is_compressed': log_file.suffix == '.gz'
                }
                log_files.append(log_info)
        
        return sorted(log_files, key=lambda x: x['modified'], reverse=True)
    
    def show_log_stats(self):
        """Show statistics about log files"""
        log_files = self.list_log_files()
        
        if not log_files:
            print("No log files found.")
            return
        
        print("\nLog Files Statistics:")
        print("=" * 80)
        print(f"{'File Name':<30} {'Size (MB)':<10} {'Modified':<20} {'Compressed'}")
        print("-" * 80)
        
        total_size = 0
        for log_info in log_files:
            compressed_marker = "✓" if log_info['is_compressed'] else ""
            print(f"{log_info['name']:<30} {log_info['size_mb']:<10.2f} "
                  f"{log_info['modified'].strftime('%Y-%m-%d %H:%M'):<20} {compressed_marker}")
            total_size += log_info['size_mb']
        
        print("-" * 80)
        print(f"Total files: {len(log_files)}")
        print(f"Total size: {total_size:.2f} MB")
    
    def tail_log(self, log_name: str, lines: int = 50):
        """Show last N lines of a log file"""
        log_files = [f for f in self.list_log_files() if log_name in f['name']]
        
        if not log_files:
            print(f"No log file found matching '{log_name}'")
            return
        
        log_file = Path(log_files[0]['path'])
        
        try:
            if log_file.suffix == '.gz':
                with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                    content_lines = f.readlines()
            else:
                with open(log_file, 'r', encoding='utf-8') as f:
                    content_lines = f.readlines()
            
            print(f"\nLast {lines} lines of {log_file.name}:")
            print("=" * 80)
            
            for line in content_lines[-lines:]:
                print(line.rstrip())
                
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    def search_logs(self, pattern: str, log_name: str = None, context_lines: int = 2):
        """Search for pattern in log files"""
        log_files = self.list_log_files()
        
        if log_name:
            log_files = [f for f in log_files if log_name in f['name']]
        
        if not log_files:
            print("No log files found matching criteria")
            return
        
        pattern_re = re.compile(pattern, re.IGNORECASE)
        total_matches = 0
        
        for log_info in log_files:
            log_file = Path(log_info['path'])
            matches = []
            
            try:
                if log_file.suffix == '.gz':
                    with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                        lines = f.readlines()
                else:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                
                for i, line in enumerate(lines):
                    if pattern_re.search(line):
                        # Get context lines
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        context = lines[start:end]
                        
                        matches.append({
                            'line_num': i + 1,
                            'line': line.strip(),
                            'context': [l.strip() for l in context]
                        })
                
                if matches:
                    print(f"\n{len(matches)} matches in {log_file.name}:")
                    print("-" * 60)
                    
                    for match in matches[:10]:  # Limit to first 10 matches per file
                        print(f"Line {match['line_num']}: {match['line']}")
                        if context_lines > 0:
                            print("Context:")
                            for ctx_line in match['context']:
                                print(f"  {ctx_line}")
                        print()
                    
                    if len(matches) > 10:
                        print(f"... and {len(matches) - 10} more matches")
                    
                    total_matches += len(matches)
                        
            except Exception as e:
                print(f"Error searching {log_file.name}: {e}")
        
        print(f"\nTotal matches found: {total_matches}")
    
    def analyze_errors(self, days: int = 7):
        """Analyze error patterns in recent logs"""
        cutoff_date = datetime.now() - timedelta(days=days)
        log_files = [f for f in self.list_log_files() if f['modified'] > cutoff_date]
        
        error_patterns = {}
        warning_patterns = {}
        
        for log_info in log_files:
            log_file = Path(log_info['path'])
            
            try:
                if log_file.suffix == '.gz':
                    with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                        lines = f.readlines()
                else:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                
                for line in lines:
                    if '| ERROR |' in line:
                        # Extract error message
                        parts = line.split('| ERROR |')
                        if len(parts) > 1:
                            error_msg = parts[1].strip()[:100]  # First 100 chars
                            error_patterns[error_msg] = error_patterns.get(error_msg, 0) + 1
                    
                    elif '| WARNING |' in line:
                        # Extract warning message
                        parts = line.split('| WARNING |')
                        if len(parts) > 1:
                            warning_msg = parts[1].strip()[:100]
                            warning_patterns[warning_msg] = warning_patterns.get(warning_msg, 0) + 1
                            
            except Exception as e:
                print(f"Error analyzing {log_file.name}: {e}")
        
        print(f"\nError Analysis (Last {days} days):")
        print("=" * 80)
        
        if error_patterns:
            print("\nTop Errors:")
            print("-" * 40)
            sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
            for error, count in sorted_errors[:10]:
                print(f"{count:>3}x: {error}")
        
        if warning_patterns:
            print("\nTop Warnings:")
            print("-" * 40)
            sorted_warnings = sorted(warning_patterns.items(), key=lambda x: x[1], reverse=True)
            for warning, count in sorted_warnings[:10]:
                print(f"{count:>3}x: {warning}")
        
        if not error_patterns and not warning_patterns:
            print("No errors or warnings found in recent logs.")
    
    def compress_old_logs(self, days: int = 7):
        """Compress log files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        log_files = [f for f in self.list_log_files() 
                    if f['modified'] < cutoff_date and not f['is_compressed']]
        
        compressed_count = 0
        
        for log_info in log_files:
            log_file = Path(log_info['path'])
            compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
            
            try:
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original file
                log_file.unlink()
                compressed_count += 1
                print(f"Compressed: {log_file.name}")
                
            except Exception as e:
                print(f"Error compressing {log_file.name}: {e}")
        
        print(f"\nCompressed {compressed_count} log files")
    
    def clean_old_logs(self, days: int = 30, confirm: bool = False):
        """Delete log files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        old_files = [f for f in self.list_log_files() if f['modified'] < cutoff_date]
        
        if not old_files:
            print(f"No log files older than {days} days found.")
            return
        
        if not confirm:
            print(f"Found {len(old_files)} log files older than {days} days:")
            for log_info in old_files:
                print(f"  {log_info['name']} ({log_info['modified'].strftime('%Y-%m-%d')})")
            print("\nUse --confirm to actually delete these files.")
            return
        
        deleted_count = 0
        for log_info in old_files:
            try:
                Path(log_info['path']).unlink()
                deleted_count += 1
                print(f"Deleted: {log_info['name']}")
            except Exception as e:
                print(f"Error deleting {log_info['name']}: {e}")
        
        print(f"\nDeleted {deleted_count} old log files")
    
    def export_logs(self, output_file: str, days: int = 7, log_types: List[str] = None):
        """Export recent logs to a single file"""
        cutoff_date = datetime.now() - timedelta(days=days)
        log_files = [f for f in self.list_log_files() if f['modified'] > cutoff_date]
        
        if log_types:
            log_files = [f for f in log_files 
                        if any(log_type in f['name'] for log_type in log_types)]
        
        exported_lines = 0
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            out_file.write(f"# ML Module Logs Export\n")
            out_file.write(f"# Generated: {datetime.now().isoformat()}\n")
            out_file.write(f"# Period: Last {days} days\n")
            out_file.write(f"# Files: {len(log_files)}\n\n")
            
            for log_info in sorted(log_files, key=lambda x: x['modified']):
                log_file = Path(log_info['path'])
                
                out_file.write(f"\n{'='*80}\n")
                out_file.write(f"FILE: {log_file.name}\n")
                out_file.write(f"MODIFIED: {log_info['modified']}\n")
                out_file.write(f"SIZE: {log_info['size_mb']:.2f} MB\n")
                out_file.write(f"{'='*80}\n\n")
                
                try:
                    if log_file.suffix == '.gz':
                        with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                            content = f.read()
                    else:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                    
                    out_file.write(content)
                    out_file.write("\n\n")
                    exported_lines += content.count('\n')
                    
                except Exception as e:
                    out_file.write(f"ERROR READING FILE: {e}\n\n")
        
        print(f"Exported {exported_lines} lines from {len(log_files)} files to {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Manage ML module log files')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List log files')
    
    # Tail command
    tail_parser = subparsers.add_parser('tail', help='Show last lines of log file')
    tail_parser.add_argument('log_name', help='Log file name (partial match)')
    tail_parser.add_argument('--lines', '-n', type=int, default=50, help='Number of lines')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search in log files')
    search_parser.add_argument('pattern', help='Search pattern (regex)')
    search_parser.add_argument('--log', help='Specific log file (partial match)')
    search_parser.add_argument('--context', '-C', type=int, default=2, help='Context lines')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze error patterns')
    analyze_parser.add_argument('--days', type=int, default=7, help='Days to analyze')
    
    # Compress command
    compress_parser = subparsers.add_parser('compress', help='Compress old log files')
    compress_parser.add_argument('--days', type=int, default=7, help='Compress files older than N days')
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Delete old log files')
    clean_parser.add_argument('--days', type=int, default=30, help='Delete files older than N days')
    clean_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export logs to file')
    export_parser.add_argument('output_file', help='Output file path')
    export_parser.add_argument('--days', type=int, default=7, help='Days to export')
    export_parser.add_argument('--types', nargs='+', help='Log types to include')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    log_manager = LogManager()
    
    if args.command == 'list':
        log_manager.show_log_stats()
    
    elif args.command == 'tail':
        log_manager.tail_log(args.log_name, args.lines)
    
    elif args.command == 'search':
        log_manager.search_logs(args.pattern, args.log, args.context)
    
    elif args.command == 'analyze':
        log_manager.analyze_errors(args.days)
    
    elif args.command == 'compress':
        log_manager.compress_old_logs(args.days)
    
    elif args.command == 'clean':
        log_manager.clean_old_logs(args.days, args.confirm)
    
    elif args.command == 'export':
        log_manager.export_logs(args.output_file, args.days, args.types)

if __name__ == "__main__":
    main()