# Power BI Dashboard — Step-by-Step

This turns the app's prediction logs into an analytics dashboard: class
distribution, confidence trends, and volume over time. Use
`app/predictions_log_sample.csv` (180 synthetic rows) to build it right now,
then repoint it at your real `predictions_log.csv` once the live app has
collected usage.

## 1. Load the data

1. Open **Power BI Desktop** (free download from Microsoft)
2. **Get Data → Text/CSV** → select `predictions_log_sample.csv`
3. Click **Transform Data** to open Power Query, then:
   - Set `timestamp` column type to **Date/Time**
   - Set `predicted_class` type to **Whole Number**
   - Set the five `prob_*` columns and `confidence` to **Decimal Number**
4. **Close & Apply**

## 2. Add a couple of calculated columns

In Power Query (or as DAX calculated columns), add:

**Date** (for time-based visuals, separate from timestamp):
```
Date = DATEVALUE([timestamp])
```

**Confidence Band** (for a quick quality-of-prediction breakdown):
```
Confidence Band =
SWITCH(
    TRUE(),
    [confidence] >= 0.85, "High (≥85%)",
    [confidence] >= 0.65, "Medium (65–85%)",
    "Low (<65%)"
)
```

## 3. Key DAX measures

```
Total Predictions = COUNTROWS('predictions_log_sample')

Avg Confidence = AVERAGE('predictions_log_sample'[confidence])

% Severe or Proliferative =
DIVIDE(
    CALCULATE(COUNTROWS('predictions_log_sample'),
        'predictions_log_sample'[predicted_label] IN {"Severe","Proliferative DR"}),
    [Total Predictions]
)

Predictions Today =
CALCULATE([Total Predictions], 'predictions_log_sample'[Date] = TODAY())
```

## 4. Recommended visuals (one Power BI page)

| Visual | Fields | Purpose |
|---|---|---|
| **Card** | `Total Predictions` | headline volume |
| **Card** | `Avg Confidence` | model calibration at a glance |
| **Card** | `% Severe or Proliferative` | clinical-risk signal |
| **Donut chart** | `predicted_label` (legend), count (values) | class distribution — will visibly skew toward "No DR", which is realistic and worth calling out |
| **Column chart** | `Confidence Band` (axis), count (values) | how often the model is confident vs. uncertain |
| **Line chart** | `Date` (axis), `Total Predictions` (values) | usage volume over time |
| **Table** | `timestamp`, `filename`, `predicted_label`, `confidence` | raw drill-through detail, sorted by timestamp descending |
| **Stacked bar** | `predicted_label` (axis), `prob_no_dr`...`prob_proliferative` (values) | average class-probability spread per predicted label — shows how "sure" the model tends to be for each class |

Arrange the three cards across the top, donut + column chart in the middle
row, line chart and table on the bottom row.

## 5. Publish it as a live/shareable dashboard

1. **Home → Publish** → sign in with a Microsoft/Power BI account → choose a workspace
2. In the **Power BI Service** (app.powerbi.com), open the published report
3. **File → Publish to web** (if you want a public embeddable link) *or*
   **Share** (if you want to share within an organization with access control)
4. You'll get a live URL you can drop into a resume/portfolio — it updates
   whenever you refresh the underlying dataset

## 6. Refreshing with real data later

Once your Streamlit app has been used for a while:
1. Download `predictions_log.csv` from the app's "Recent predictions" panel
2. In Power BI Desktop: **Home → Transform Data → Data Source Settings** →
   point it at the new file → **Refresh**
3. Republish (step 5) to update the live dashboard

For a fully automated refresh (no manual re-upload), the log would need to
live somewhere Power BI can poll on a schedule — e.g. write
`predictions_log.csv` to a OneDrive/SharePoint folder from the app instead
of local disk, then set up a **scheduled refresh** in the Power BI Service
pointed at that cloud file.
