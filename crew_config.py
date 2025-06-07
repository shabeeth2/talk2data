"""
CrewAI Configuration for Insurance Analytics Platform
"""
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

load_dotenv()

# Set environment variables for CrewAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "dummy-key")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Use CrewAI's LLM class with Google Gemini
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=google_api_key
)

# Database Schema for Insurance
INSURANCE_SCHEMA = """
Insurance Database Schema (newSynthetic70k.db):
1. customers (cust_id, name, email, phone, address_id, agent_id, joined_date, status)
2. policies (policy_id, customer_id, policy_type, start_date, end_date, premium_amount, status)
3. claims (claim_id, policy_id, claim_date, claim_amount, claim_status, approved_amount)
4. agents (agent_id, agent_name, email, phone, hire_date, commission_rate)
5. commissions (commission_id, agent_id, policy_id, commission_amount, paid_date)
6. addresses (address_id, street, city, county, postcode, country)
7. prospects (prospect_id, name, email, phone, created_at, status)
8. quotes (quote_id, prospect_id, quote_date, premium_amount, valid_till, status)

Key Relationships:
- customers.address_id -> addresses.address_id
- customers.agent_id -> agents.agent_id
- policies.customer_id -> customers.cust_id
- claims.policy_id -> policies.policy_id
- commissions.agent_id -> agents.agent_id
- commissions.policy_id -> policies.policy_id
- quotes.prospect_id -> prospects.prospect_id

Note: Use column names exactly as specified above for accurate queries.
"""

# Pydantic Models for Data Structures
class ToolInvocationPlan(BaseModel):
    tool_name: str
    invoke_count: int
    execution_order: int
    required: bool
    conditional: Optional[str] = None
    dependencies: List[str] = []

class SQLExecutionStrategy(BaseModel):
    execution_type: str  # "simple", "multi-fact", "comparative", "conditional"
    sql_calls_needed: int
    execution_sequence: List[str]
    merge_strategy: Optional[str] = None

class OrchestrationPlan(BaseModel):
    user_intent: str
    confidence_score: float
    tool_invocation_sequence: List[ToolInvocationPlan]
    sql_strategy: SQLExecutionStrategy
    skip_decisions: Dict[str, bool]
    conditional_logic: Dict[str, str]
    execution_metadata: Dict[str, Any]

class AnalyticalPlan(BaseModel):
    intent: str
    filters: List[str]
    metrics: List[str]
    group_by: List[str]
    time_dimension: Optional[str] = None
    aggregation_type: str = "SUM"
    orchestration_plan: OrchestrationPlan

class SQLQuery(BaseModel):
    query: str
    explanation: str
    estimated_result_type: str

class VisualizationMetadata(BaseModel):
    chart_type: str
    title: str
    x_axis: str
    y_axis: str
    color_scheme: Optional[str] = None
    additional_config: Dict[str, Any] = {}

class AnalyticsResponse(BaseModel):
    natural_language_summary: str
    visualization_directive: VisualizationMetadata
    key_insights: List[str]
    data_context: Dict[str, Any]

# Chart Type Mapping
CHART_TYPE_MAPPING = {
    "count": "bar",
    "comparison": "bar", 
    "trend": "line",
    "distribution": "pie",
    "correlation": "scatter",
    "time_series": "line",
    "geographic": "map",
    "ranking": "bar"
}

# Configuration Constants
CREW_CONFIG = {
    "verbose": True,
    "memory": True,
    "max_iter": 3,
    "max_execution_time": 30
}