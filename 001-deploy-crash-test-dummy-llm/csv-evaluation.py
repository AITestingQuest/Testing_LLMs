import pandas as pd

# Load the CSV file
file_path = 'llm-test-results-evaluated.csv'
data = pd.read_csv(file_path)

# Strip any leading/trailing whitespace from the column names
data.columns = data.columns.str.strip()

# Group by model and evaluation, then count occurrences
summary = data.groupby(['Model', 'Evaluation']).size().unstack(fill_value=0)

# Calculate pass/fail rates
summary['PASS_RATE'] = summary['PASS'] / (summary['PASS'] + summary['FAIL'])

# Save the summary to a new CSV file
summary_file_path = 'llm-test-results-summary.csv'
summary.to_csv(summary_file_path)

print(summary)
