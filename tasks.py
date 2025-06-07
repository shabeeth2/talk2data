"""
CrewAI Tasks for Insurance Analytics Platform
"""
from crewai import Task
from crew_config import AnalyticalPlan, SQLQuery, VisualizationMetadata, AnalyticsResponse
import json

class AnalyticsTasks:
    """Factory class for creating specialized analytics tasks"""
    
    @staticmethod
    def create_planning_task(user_query: str, planner_agent):
        """Creates an enhanced orchestration planning task for the Planner Agent"""
        return Task(
            description=f"""
            Analyze the following user query and create a comprehensive orchestration plan for insurance data analysis:
            
            User Query: "{user_query}"
            
            Your task is to create a COMPLETE ORCHESTRATION PLAN that includes:
            
            1. INTENT ANALYSIS:
               - Decode the user's intent and assign a confidence score (0.0-1.0)
               - Identify query complexity level (simple, moderate, complex)
               - Determine if this is a follow-up question or new request
            
            2. TOOL ORCHESTRATION STRATEGY:
               - Decide which agents to invoke: sql_generator, visualizer, response_composer
               - Determine invoke count for each agent (1-3 times max)
               - Set execution order and dependencies
               - Mark which tools are required vs optional
            
            3. SQL EXECUTION STRATEGY:
               - Choose execution type: "simple", "multi-fact", "comparative", "conditional"
               - Define number of SQL calls needed (1-5 max)
               - Specify execution sequence if multiple calls
               - Set merge strategy if combining results
            
            4. SKIP LOGIC DECISIONS:
               - Skip SQL if: cached context available, non-data question, previous result usable
               - Skip Visualization if: user wants numbers only, text summary requested, external pipe
               - Skip Response if: data export only, intermediate step, pipeline mode
            
            5. CONDITIONAL LOGIC:
               - Define threshold-based execution paths
               - Set confidence-based tool invocation
               - Create result-dependent next steps
            
            Return a complete JSON object with this exact structure:
            {{
                "intent": "Clear description of user's analytical goal",
                "filters": ["list of data filters to apply"],
                "metrics": ["list of metrics to calculate"],
                "group_by": ["list of dimensions to group by"],
                "time_dimension": "time field if temporal analysis needed",
                "aggregation_type": "SUM/COUNT/AVG/MAX/MIN",
                "orchestration_plan": {{
                    "user_intent": "Refined intent description",
                    "confidence_score": 0.95,
                    "tool_invocation_sequence": [
                        {{
                            "tool_name": "sql_generator",
                            "invoke_count": 1,
                            "execution_order": 1,
                            "required": true,
                            "conditional": null,
                            "dependencies": []
                        }},
                        {{
                            "tool_name": "visualizer",
                            "invoke_count": 1,
                            "execution_order": 2,
                            "required": true,
                            "conditional": "if data_rows > 0",
                            "dependencies": ["sql_generator"]
                        }},
                        {{
                            "tool_name": "response_composer",
                            "invoke_count": 1,
                            "execution_order": 3,
                            "required": true,
                            "conditional": null,
                            "dependencies": ["sql_generator", "visualizer"]
                        }}
                    ],
                    "sql_strategy": {{
                        "execution_type": "simple",
                        "sql_calls_needed": 1,
                        "execution_sequence": ["main_query"],
                        "merge_strategy": null
                    }},
                    "skip_decisions": {{
                        "skip_sql": false,
                        "skip_visualization": false,
                        "skip_response": false
                    }},
                    "conditional_logic": {{
                        "confidence_threshold": "if confidence > 0.8 then proceed",
                        "data_threshold": "if result_count > 0 then visualize",
                        "error_handling": "if sql_error then use fallback"
                    }},
                    "execution_metadata": {{
                        "query_type": "analytical",
                        "expected_duration": "fast",
                        "resource_requirements": "low"
                    }}
                }}
            }}
            
            EXECUTION PATTERN EXAMPLES:
            - "Show total claims" → Simple: SQL(1) → Viz(1) → Response(1)
            - "Compare health vs auto claims" → Multi-fact: SQL(2) → Merge → Viz(1) → Response(1)
            - "Claims before and after 2024" → Comparative: SQL(2) → Compare → Viz(1) → Response(1)
            - "Just give me the claim count" → Summary: SQL(1) → Skip Viz → Response(1)
            - "What was that chart about?" → Cached: Skip SQL → Use Previous → Response(1)
            """,
            agent=planner_agent,
            expected_output="A comprehensive JSON orchestration plan with intent analysis, tool sequencing, SQL strategy, skip logic, and conditional execution paths."
        )
    
    @staticmethod
    def create_sql_generation_task(analytical_plan: str, sql_agent):
        """Creates a task for the SQL Generator Agent to translate plan into SQL"""
        return Task(
            description=f"""
            Based on the following analytical plan, generate an efficient SQL query for the insurance database:
            
            Analytical Plan: {analytical_plan}
            
            Your task is to:
            1. Translate the analytical plan into a valid SQL query
            2. Use proper JOINs to connect related tables
            3. Apply appropriate filters and grouping
            4. Ensure the query is optimized and follows best practices
            5. Include proper column aliases for clarity
            6. Handle date/time formatting if needed
            
            Database tables available:
            - customer, policy, claim, agent, sales, commission, address, quote
            
            Return your response as a JSON object with:
            - query: the complete SQL query
            - explanation: brief explanation of what the query does
            - estimated_result_type: description of expected result structure
            
            Example output:
            {{
                "query": "SELECT p.policy_type, SUM(c.claim_amount) as total_claims FROM policy p JOIN claim c ON p.policy_id = c.policy_id WHERE c.claim_status = 'Approved' GROUP BY p.policy_type",
                "explanation": "Sums approved claim amounts grouped by policy type",
                "estimated_result_type": "Two columns: policy_type and total_claims"
            }}
            """,
            agent=sql_agent,
            expected_output="A JSON object with the SQL query, explanation, and estimated result type."
        )
    
    @staticmethod
    def create_visualization_task(sql_query_info: str, user_intent: str, viz_agent):
        """Creates a task for the Visualization Agent to determine chart type and metadata"""
        return Task(
            description=f"""
            Based on the SQL query and user intent, determine the optimal visualization approach:
            
            SQL Query Information: {sql_query_info}
            User Intent: {user_intent}
            
            Your task is to:
            1. Analyze the expected data structure and user intent
            2. Select the most appropriate chart type (bar, line, pie, scatter, map)
            3. Define chart configuration including titles and axis labels
            4. Suggest color schemes if relevant
            5. Provide any additional chart configuration
            
            Chart type guidelines:
            - Use 'bar' for comparisons and rankings
            - Use 'line' for trends over time
            - Use 'pie' for distributions and parts-of-whole
            - Use 'scatter' for correlations
            - Use 'map' for geographic data
            
            Return your response as a JSON object with:
            - chart_type: the recommended chart type
            - title: descriptive chart title
            - x_axis: label for x-axis
            - y_axis: label for y-axis
            - color_scheme: optional color scheme
            - additional_config: any extra configuration needed
            
            Example output:
            {{
                "chart_type": "bar",
                "title": "Total Claim Amounts by Policy Type",
                "x_axis": "Policy Type",
                "y_axis": "Total Claim Amount ($)",
                "color_scheme": "insurance_blue",
                "additional_config": {{"horizontal": false, "show_values": true}}
            }}
            """,
            agent=viz_agent,
            expected_output="A JSON object with chart type, title, axis labels, and configuration."
        )
    
    @staticmethod
    def create_response_composition_task(sql_results: str, viz_metadata: str, user_query: str, response_agent):
        """Creates a task for the Response Generator to compose final analytics response"""
        return Task(
            description=f"""
            Compose a comprehensive analytics response based on the following information:
            
            Original User Query: {user_query}
            SQL Results: {sql_results}
            Visualization Metadata: {viz_metadata}
            
            Your task is to:
            1. Write a clear natural language summary of the findings
            2. Extract key business insights from the data
            3. Provide actionable recommendations where appropriate
            4. Include the visualization directive for the frontend
            5. Add relevant context about the data and analysis
            
            Return your response as a JSON object with:
            - natural_language_summary: clear explanation of the results
            - visualization_directive: the visualization metadata
            - key_insights: list of important findings
            - data_context: additional context about the analysis
            
            Example output:
            {{
                "natural_language_summary": "The analysis shows that Health insurance policies generate the highest claim amounts...",
                "visualization_directive": {{"chart_type": "bar", "title": "...", ...}},
                "key_insights": ["Health policies account for 60% of total claims", "Life policies have lower claim frequency"],
                "data_context": {{"total_records": 1250, "time_period": "All time", "data_quality": "Complete"}}
            }}
            """,
            agent=response_agent,
            expected_output="A JSON object with natural language summary, visualization directive, key insights, and data context."
        )