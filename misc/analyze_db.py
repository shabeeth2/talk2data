import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

def analyze_database():
    """Analyze the current database structure and data quality"""
    conn = sqlite3.connect('data/newSynthetic70k.db')
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    print("=== DATABASE ANALYSIS ===")
    print(f"Available tables: {tables}")
    print()
    
    table_info = {}
    
    for table_name in tables:
        print(f"Table: {table_name}")
        print("-" * 40)
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        
        table_info[table_name] = {
            'columns': columns,
            'row_count': row_count,
            'sample_data': sample_data
        }
        
        print(f"Columns ({len(columns)}):")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        print(f"Row count: {row_count}")
        print("Sample data:")
        for i, row in enumerate(sample_data[:3]):
            print(f"  Row {i+1}: {row}")
        print()
    
    conn.close()
    return table_info

if __name__ == "__main__":
    analyze_database()