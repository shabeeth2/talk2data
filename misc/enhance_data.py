import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

def enhance_database_for_visualization():
    """Add quality data to improve visualization capabilities with Life and Health policy types"""
    
    conn = sqlite3.connect('data/newSynthetic70k.db')
    cursor = conn.cursor()
    
    print("=== ENHANCING DATABASE FOR BETTER VISUALIZATIONS ===")
    
    # 1. Check current data and enhance sales table
    print("1. Enhancing sales data with realistic patterns...")
    
    # Get existing policies (only Life and Health)
    cursor.execute("SELECT policy_id, customer_id, premium_amount, policy_type FROM policies WHERE policy_type IN ('Life', 'Health') LIMIT 10000")
    policies = cursor.fetchall()
    
    # Get agents for assignment
    cursor.execute("SELECT agent_id FROM agents LIMIT 200")
    agents = [row[0] for row in cursor.fetchall()]
    
    # Get customer addresses for region mapping
    cursor.execute('''
        SELECT c.cust_id, a.county 
        FROM customers c 
        JOIN addresses a ON c.address_id = a.address_id 
        LIMIT 10000
    ''')
    customer_regions = dict(cursor.fetchall())
    
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Clear existing sales data to rebuild with patterns
    cursor.execute("DELETE FROM sales")
    
    sales_data = []
    
    # Generate 3 years of sales data (2022-2025)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2025, 7, 24)  # Current date
    
    current_date = start_date
    while current_date <= end_date:
        # Seasonal patterns differ for Life vs Health
        month = current_date.month
        
        # Health insurance: Higher enrollment in Q4 (open enrollment) and Q1
        health_multiplier = 1.0
        if month in [10, 11, 12, 1]:  # Open enrollment season
            health_multiplier = 1.6
        elif month in [6, 7, 8]:  # Summer
            health_multiplier = 0.7
        
        # Life insurance: More consistent but higher in Q1 (New Year resolutions) and Q4 (estate planning)
        life_multiplier = 1.0
        if month in [1, 2, 11, 12]:  # New Year and estate planning season
            life_multiplier = 1.3
        elif month in [6, 7]:  # Summer vacation season
            life_multiplier = 0.8
        
        # Generate sales for this month (sample from policies)
        monthly_policies = random.sample(policies, min(len(policies), 200))
        
        for policy_id, customer_id, premium_amount, policy_type in monthly_policies:
            # Apply seasonal multiplier based on policy type
            multiplier = health_multiplier if policy_type == 'Health' else life_multiplier
            
            # Probability of sale based on season and random factors
            if random.random() < (multiplier * 0.3):  # 30% base probability adjusted by season
                sale_id = f"SA{random.randint(10000000, 99999999)}"
                
                # Add variance to dates within the month
                days_in_month = min(27, (end_date - current_date).days)
                if days_in_month > 0:
                    sale_date = current_date + timedelta(days=random.randint(0, days_in_month))
                else:
                    sale_date = current_date
                
                region = customer_regions.get(customer_id, random.choice(regions))
                agent_id = random.choice(agents)
                
                # Commission rates differ: Health 8-12%, Life 10-15%
                if policy_type == 'Health':
                    commission_rate = random.uniform(0.08, 0.12)
                else:  # Life
                    commission_rate = random.uniform(0.10, 0.15)
                
                commission = premium_amount * commission_rate
                
                sales_data.append((
                    sale_id,
                    policy_id,
                    agent_id,
                    customer_id,
                    sale_date.strftime('%Y-%m-%d'),
                    premium_amount,
                    round(commission, 2),
                    region
                ))
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    # Insert new sales data
    cursor.executemany('''
        INSERT INTO sales 
        (sale_id, policy_id, agent_id, customer_id, sale_date, premium_amount, commission_amount, region)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', sales_data)
    
    print(f"   Generated {len(sales_data)} sales records with seasonal patterns")
    
    # 2. Enhance customer demographics
    print("2. Enhancing customer demographics with Life vs Health preferences...")
    
    # Clear existing demographics
    cursor.execute("DELETE FROM customer_demographics")
    
    cursor.execute("SELECT cust_id FROM customers LIMIT 15000")
    customers = [row[0] for row in cursor.fetchall()]
    
    # Check if they have Life or Health policies
    cursor.execute('''
        SELECT DISTINCT customer_id, policy_type 
        FROM policies 
        WHERE policy_type IN ('Life', 'Health')
    ''')
    customer_policy_types = {}
    for customer_id, policy_type in cursor.fetchall():
        if customer_id not in customer_policy_types:
            customer_policy_types[customer_id] = []
        customer_policy_types[customer_id].append(policy_type)
    
    demographics_data = []
    income_brackets = ['<30k', '30k-50k', '50k-75k', '75k-100k', '100k-150k', '>150k']
    occupations = ['Professional', 'Healthcare', 'Education', 'Technology', 'Sales', 'Management', 'Skilled Labor', 'Service']
    
    # Use a set to ensure unique demo_ids
    used_demo_ids = set()
    
    for i, customer_id in enumerate(customers):
        # Generate unique demo_id
        demo_id = f"DM{str(i+1).zfill(8)}"
        while demo_id in used_demo_ids:
            demo_id = f"DM{random.randint(10000000, 99999999)}"
        used_demo_ids.add(demo_id)
        
        # Demographics that influence policy choice
        policy_types = customer_policy_types.get(customer_id, [])
        
        if 'Life' in policy_types:
            # Life insurance holders tend to be older, married with families
            age = random.randint(30, 55)
            family_size = random.randint(2, 5)
            income_bias = random.choices(income_brackets, weights=[5, 10, 20, 25, 25, 15])[0]
            health_score = random.randint(6, 9)
        else:
            # Health insurance holders more diverse
            age = random.randint(25, 65)
            family_size = random.randint(1, 4)
            income_bias = random.choices(income_brackets, weights=[10, 15, 20, 20, 20, 15])[0]
            health_score = random.randint(4, 8)
        
        demographics_data.append((
            demo_id,
            customer_id,
            age,
            random.choice(['Male', 'Female']),
            income_bias,
            random.choice(occupations),
            family_size,
            health_score
        ))
    
    cursor.executemany('''
        INSERT INTO customer_demographics 
        (demo_id, customer_id, age, gender, income_bracket, occupation_category, family_size, health_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', demographics_data)
    
    print(f"   Generated {len(demographics_data)} demographic records")
    
    # 3. Ensure policy types are only Life and Health
    print("3. Standardizing policy types to Life and Health only...")
    cursor.execute('''
        UPDATE policies 
        SET policy_type = CASE 
            WHEN RANDOM() % 2 = 0 THEN 'Life'
            ELSE 'Health'
        END
        WHERE policy_type NOT IN ('Life', 'Health')
    ''')
    
    # 4. Fix any date inconsistencies
    print("4. Fixing date inconsistencies...")
    cursor.execute('''
        UPDATE policies 
        SET end_date = date(start_date, '+1 year')
        WHERE date(end_date) < date(start_date)
    ''')
    
    # 5. Create aggregated views for better visualization
    print("5. Creating helpful views for visualization...")
    
    # Drop existing views if they exist
    cursor.execute("DROP VIEW IF EXISTS monthly_sales_summary")
    cursor.execute("DROP VIEW IF EXISTS policy_type_performance")
    cursor.execute("DROP VIEW IF EXISTS agent_performance_by_type")
    cursor.execute("DROP VIEW IF EXISTS regional_performance")
    
    # Monthly sales summary
    cursor.execute('''
        CREATE VIEW monthly_sales_summary AS
        SELECT 
            strftime('%Y-%m', sale_date) as month,
            COUNT(*) as total_sales,
            SUM(premium_amount) as total_premium,
            SUM(commission_amount) as total_commission,
            AVG(premium_amount) as avg_premium
        FROM sales
        GROUP BY strftime('%Y-%m', sale_date)
        ORDER BY month
    ''')
    
    # Policy type performance
    cursor.execute('''
        CREATE VIEW policy_type_performance AS
        SELECT 
            p.policy_type,
            COUNT(s.sale_id) as sales_count,
            SUM(s.premium_amount) as total_revenue,
            AVG(s.premium_amount) as avg_premium,
            COUNT(DISTINCT s.agent_id) as agents_involved
        FROM policies p
        JOIN sales s ON p.policy_id = s.policy_id
        GROUP BY p.policy_type
    ''')
    
    # Agent performance by policy type
    cursor.execute('''
        CREATE VIEW agent_performance_by_type AS
        SELECT 
            a.agent_id,
            a.agent_name,
            p.policy_type,
            COUNT(s.sale_id) as policies_sold,
            SUM(s.commission_amount) as total_commission,
            AVG(s.premium_amount) as avg_deal_size
        FROM agents a
        JOIN sales s ON a.agent_id = s.agent_id
        JOIN policies p ON s.policy_id = p.policy_id
        GROUP BY a.agent_id, p.policy_type
    ''')
    
    # Regional performance
    cursor.execute('''
        CREATE VIEW regional_performance AS
        SELECT 
            s.region,
            p.policy_type,
            COUNT(s.sale_id) as sales_count,
            SUM(s.premium_amount) as total_revenue,
            AVG(s.premium_amount) as avg_premium
        FROM sales s
        JOIN policies p ON s.policy_id = p.policy_id
        GROUP BY s.region, p.policy_type
        ORDER BY total_revenue DESC
    ''')
    
    conn.commit()
    conn.close()
    
    print("\n=== DATABASE ENHANCEMENT COMPLETE ===")
    print("Enhanced for Life vs Health insurance analysis:")
    print("✓ Sales data with realistic seasonal patterns (2022-2025)")
    print("✓ Customer demographics linked to policy preferences")
    print("✓ Regional sales distribution")
    print("✓ Agent performance metrics")
    print("✓ Monthly/yearly trend data")
    print("✓ Policy types standardized to Life and Health only")
    print("✓ Created helpful views for visualization")
    print("\nNew views available for queries:")
    print("- monthly_sales_summary: Time series analysis")
    print("- policy_type_performance: Life vs Health comparison")
    print("- agent_performance_by_type: Agent rankings by policy type")
    print("- regional_performance: Geographic sales analysis")
    print("\nReady for advanced visualizations!")

if __name__ == "__main__":
    enhance_database_for_visualization()