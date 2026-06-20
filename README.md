# Salesforce Opportunity Win-Loss Analysis

This project analyzes a public CRM opportunity dataset and builds a logistic regression model to study what is associated with closed-won versus closed-lost deals.

## Files

- `salesforce_opportunity_win_loss_analysis.py` - main Python analysis script
- `salesforce_opportunity_win_loss_analysis.ipynb` - notebook version of the analysis
- `requirements.txt` - Python dependencies
- `data/` - cached public source files
- `outputs/` - generated tables, plots, and model artifacts

## Public dataset

I used the public CRM Sales Opportunities tables mirrored in this GitHub repository:

- `sales_pipeline.csv`
- `accounts.csv`
- `sales_teams.csv`
- `products.csv`

Source mirror:

- https://github.com/DiogoSoares3/CRM-AI-Analysis/tree/main/data

Original dataset description:

- https://mavenanalytics.io/data-playground/crm-sales-opportunities

## How to run

```bash
pip install -r requirements.txt
python salesforce_opportunity_win_loss_analysis.py
```

## Notes

- The public opportunity table does not expose a true `LeadSource` field.
- It also exposes realized `close_value`, which would leak the target because lost deals carry zero booked revenue.
- To keep the model fair, the script uses product list price as a public proxy for deal size and focuses on available CRM-style predictors such as product, industry, region, company size, and engagement timing.
