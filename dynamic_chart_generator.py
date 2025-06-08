import json

def generate_streamlit_chart_code(spec):
    """
    Generate a Streamlit code snippet for a chart based on the provided spec.
    spec should include:
      - chart_type: 'bar', 'line', or 'scatter'
      - data_path: path to a CSV file (optional if using existing df)
      - x: name of x-axis column
      - y: name of y-axis column
      - title: chart title (optional)
      - width: int (optional)
      - height: int (optional)
    Returns a string containing the code.
    """
    chart_type = spec.get('chart_type', 'bar')
    data_path = spec.get('data_path')
    x = spec['x']
    y = spec['y']
    title = spec.get('title', '')
    width = spec.get('width', 800)
    height = spec.get('height', 600)

    lines = []
    lines.append('import streamlit as st')
    lines.append('import pandas as pd')
    lines.append('import plotly.express as px')
    lines.append('')
    if data_path:
        lines.append(f"df = pd.read_csv(r'{data_path}')")
    else:
        lines.append('# df should be a pandas DataFrame already loaded')
    lines.append('')
    lines.append(f"# Create {chart_type} chart")
    if chart_type == 'bar':
        lines.append(f"fig = px.bar(df, x='{x}', y='{y}', title='{title}', width={width}, height={height})")
    elif chart_type == 'line':
        lines.append(f"fig = px.line(df, x='{x}', y='{y}', title='{title}', width={width}, height={height})")
    elif chart_type == 'scatter':
        lines.append(f"fig = px.scatter(df, x='{x}', y='{y}', title='{title}', width={width}, height={height})")
    else:
        lines.append(f"# Unsupported chart type: {chart_type}")
        lines.append('fig = None')
    lines.append('')
    lines.append('if fig is not None:')
    lines.append('    st.plotly_chart(fig, use_container_width=True)')

    return '\n'.join(lines)


def main():
    print('Enter chart spec as JSON:')
    raw = input()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError:
        print('Invalid JSON spec.')
        return

    code = generate_streamlit_chart_code(spec)
    print('\nGenerated Streamlit Chart Code:\n')
    print(code)


if __name__ == '__main__':
    main()