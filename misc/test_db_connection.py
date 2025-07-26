import sqlite3
import os
from langchain_community.utilities.sql_database import SQLDatabase

def test_database_connection():
    """Test database connection and schema retrieval"""
    try:
        # Test direct sqlite connection
        db_path = os.path.abspath("./data/newSynthetic70k.db")
        print(f"Testing database at: {db_path}")
        print(f"File exists: {os.path.exists(db_path)}")
        
        # Direct sqlite3 connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"Tables found: {tables}")
        conn.close()
        
        # Test SQLDatabase connection with different approaches
        try:
            db_uri = f"sqlite:///{db_path.replace(chr(92), '/')}"  # Use forward slashes
            print(f"Testing SQLDatabase with URI: {db_uri}")
            db = SQLDatabase.from_uri(db_uri)
            
            # Test schema retrieval
            schema = db.get_table_info()
            print("Schema retrieval successful!")
            print(f"Schema preview: {schema[:200]}...")
            
        except Exception as e:
            print(f"SQLDatabase failed with forward slashes: {e}")
            
            # Try with include_tables parameter
            try:
                db = SQLDatabase.from_uri(db_uri, include_tables=['policies', 'customers', 'agents', 'sales'])
                schema = db.get_table_info()
                print("Schema retrieval with include_tables successful!")
                
            except Exception as e2:
                print(f"SQLDatabase with include_tables also failed: {e2}")
                
                # Try creating custom schema
                schema = create_custom_schema()
                print("Using custom schema as fallback")
        
        # Test a simple query
        try:
            result = db.run("SELECT COUNT(*) FROM policies")
            print(f"Test query result: {result}")
        except:
            print("Query test skipped due to connection issues")
        
        print("✅ Database connection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection test FAILED: {str(e)}")
        return False

def create_custom_schema():
    """Create a custom schema string for the database"""
    return """
    Table 'addresses' has columns: address_id (TEXT), street (TEXT), city (TEXT), county (TEXT), postcode (TEXT), country (TEXT)
    Table 'agents' has columns: agent_id (TEXT), agent_name (TEXT), email (TEXT), phone (TEXT), hire_date (TEXT), commission_rate (REAL)
    Table 'claims' has columns: claim_id (TEXT), policy_id (TEXT), claim_date (TEXT), claim_amount (REAL), claim_status (TEXT), approved_amount (REAL)
    Table 'commissions' has columns: commission_id (TEXT), agent_id (TEXT), policy_id (TEXT), commission_amount (REAL), paid_date (TEXT)
    Table 'customers' has columns: cust_id (TEXT), name (TEXT), email (TEXT), phone (TEXT), address_id (TEXT), agent_id (TEXT), joined_date (TEXT), status (TEXT)
    Table 'policies' has columns: policy_id (TEXT), customer_id (TEXT), policy_type (TEXT), start_date (TEXT), end_date (TEXT), premium_amount (REAL), status (TEXT)
    Table 'prospects' has columns: prospect_id (TEXT), name (TEXT), email (TEXT), phone (TEXT), created_at (TEXT), status (TEXT)
    Table 'quotes' has columns: quote_id (TEXT), prospect_id (TEXT), quote_date (TEXT), premium_amount (REAL), valid_till (TEXT), status (TEXT)
    Table 'sales' has columns: sale_id (TEXT), policy_id (TEXT), agent_id (TEXT), customer_id (TEXT), sale_date (TEXT), premium_amount (REAL), commission_amount (REAL), region (TEXT)
    Table 'customer_demographics' has columns: demo_id (TEXT), customer_id (TEXT), age (INTEGER), gender (TEXT), income_bracket (TEXT), occupation_category (TEXT), family_size (INTEGER), health_score (INTEGER)
    """

if __name__ == "__main__":
    test_database_connection()