# 🤖 Agentic Conversational Analytics Platform

A sophisticated multi-agent system built with **CrewAI** for natural language querying of insurance databases. The platform employs four specialized AI agents that work together to convert natural language questions into SQL queries, execute them, and generate visualizations with business insights.

## 🏗️ System Architecture

### Four Specialized Agents

1. **🧠 Planner Agent** - Converts natural language queries into structured analytical plans
   - Analyzes user intent and requirements
   - Breaks down complex requests into structured components
   - Identifies metrics, filters, groupings, and time dimensions

2. **🔍 SQL Generator Agent** - Translates analytical plans into optimized SQL queries
   - Expert knowledge of insurance database schema
   - Generates efficient, performant SQL queries
   - Follows SQL best practices and optimization techniques

3. **🎨 Visualization Agent** - Determines optimal chart types and visualization metadata
   - Analyzes data structure and user intent
   - Recommends appropriate chart types (bar, line, pie, scatter, map)
   - Generates complete visualization configuration

4. **📝 Response Composer Agent** - Creates comprehensive natural language summaries
   - Transforms technical results into business insights
   - Provides actionable recommendations
   - Generates key insights and data context

### 🔄 Workflow Process

```
User Query → Planner Agent → SQL Generator → Query Execution → Visualization Agent → Response Composer → Final Output
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key
- SQLite database with insurance data

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd talk2data
pip install -r requirements.txt
```

2. **Configure environment variables:**
Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

3. **Run the application:**
```bash
streamlit run app.py
```

## 📊 Features

### Natural Language Querying
Ask questions in plain English:
- "Show total claims by policy type"
- "What's the premium trend over the last 6 months?"
- "Which agents have the highest sales performance?"
- "Display customer distribution by state"

### Intelligent Visualization
- Automatic chart type selection based on data and intent
- Interactive Plotly visualizations
- Customized titles, labels, and styling

### Business Intelligence
- Key insights extraction
- Actionable recommendations
- Data context and quality information
- Technical details for power users

## 🗄️ Database Schema

The system works with the following insurance database tables:

- **customer** - Customer information and demographics
- **policy** - Insurance policies and coverage details
- **claim** - Insurance claims and processing status
- **agent** - Sales agents and performance data
- **sales** - Sales transactions and commissions
- **commission** - Commission payments and rates
- **address** - Geographic information
- **quote** - Insurance quotes and pricing

## 🔧 Technical Implementation

### Core Components

1. **`crew_config.py`** - Configuration, data models, and LLM setup
2. **`agents.py`** - CrewAI agent definitions and specializations
3. **`tasks.py`** - Task definitions for each agent
4. **`app.py`** - Main Streamlit application and workflow orchestration

### Key Technologies

- **CrewAI** - Multi-agent orchestration framework
- **LangChain** - LLM integration and tooling
- **Google Gemini** - Large language model for AI agents
- **Streamlit** - Web application framework
- **Plotly** - Interactive visualization library
- **SQLite** - Database for insurance data

### Data Models

```python
class AnalyticalPlan(BaseModel):
    intent: str
    filters: List[str]
    metrics: List[str]
    group_by: List[str]
    time_dimension: Optional[str]
    aggregation_type: str

class VisualizationMetadata(BaseModel):
    chart_type: str
    title: str
    x_axis: str
    y_axis: str
    color_scheme: Optional[str]
    additional_config: Dict[str, Any]
```

## 🎯 Example Workflows

### Workflow 1: Claims Analysis
```
User: "Show me total claims by policy type"
↓
Planner: Creates plan to sum claim amounts grouped by policy type
↓
SQL Generator: "SELECT policy_type, SUM(claim_amount) FROM policy p JOIN claim c..."
↓
Visualizer: Recommends bar chart with policy types on x-axis
↓
Response Composer: "Auto insurance generates the highest claims at $3.2M..."
```

### Workflow 2: Trend Analysis
```
User: "What's the premium trend over time?"
↓
Planner: Identifies time-series analysis with premium aggregation
↓
SQL Generator: Creates query with date grouping and SUM
↓
Visualizer: Recommends line chart for time series
↓
Response Composer: "Premium collection shows 15% growth over 6 months..."
```

## 🔍 Advanced Features

### Mock Data Integration
For demonstration purposes, the system includes intelligent mock data generation:
- Context-aware dataset selection based on query intent
- Realistic insurance industry data patterns
- Seamless integration with real database queries

### Error Handling
- Graceful fallbacks for JSON parsing errors
- Database connection error handling
- Visualization rendering error recovery

### Extensibility
- Modular agent design for easy customization
- Configurable chart type mappings
- Expandable database schema support

## 🎨 User Interface

### Main Dashboard
- Clean, intuitive query interface
- Real-time workflow progress tracking
- Interactive visualizations
- Comprehensive results display

### Sidebar Features
- Active agent status indicators
- Sample query buttons for quick testing
- Agent role descriptions

### Results Layout
- Two-column layout with visualization and insights
- Expandable technical details
- Raw data access
- Export capabilities

## 🔒 Security & Configuration

### Environment Variables
- `GOOGLE_API_KEY` - Google Gemini API authentication
- Database connection strings
- Optional configuration overrides

### Agent Configuration
```python
CREW_CONFIG = {
    "verbose": True,
    "memory": True,
    "max_iter": 3,
    "max_execution_time": 30
}
```

## 📈 Performance Optimization

### SQL Query Optimization
- Proper JOIN usage
- Index-aware query construction
- Subquery minimization
- Performance hints integration

### Agent Efficiency
- Limited iteration counts
- Focused task definitions
- Memory management
- Timeout controls

## 🧪 Testing & Development

### Sample Queries for Testing
1. **Claims Analysis**: "Show total claims by policy type"
2. **Trend Analysis**: "What's the premium trend over the last 6 months?"
3. **Performance Metrics**: "Which agents have the highest sales performance?"
4. **Geographic Distribution**: "Display customer distribution by state"

### Development Mode
- Enable verbose logging for agent interactions
- Technical details expansion for debugging
- JSON output inspection
- SQL query validation

## 🔮 Future Enhancements

### Planned Features
- Real-time database connection with actual insurance data
- Advanced chart types (heatmaps, treemaps, sankey diagrams)
- Multi-database support (PostgreSQL, MySQL, Oracle)
- Custom dashboard creation and saving
- Export functionality (PDF, Excel, PowerPoint)
- Natural language insights generation
- Automated report scheduling

### Integration Possibilities
- BI tool connectors (Tableau, Power BI)
- API endpoints for external consumption
- Slack/Teams integration for conversational analytics
- Email report automation
- Mobile responsive design

## 📝 License

This project is built for educational and demonstration purposes. Please ensure compliance with your organization's data governance and security policies when adapting for production use.

## 🤝 Contributing

The modular architecture makes it easy to contribute:
- Add new agent types for specialized analytics
- Extend database schema support
- Implement additional visualization types
- Enhance natural language processing capabilities

---

*Built with ❤️ using CrewAI, LangChain, and Streamlit*

