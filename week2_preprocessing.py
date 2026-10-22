"""Example data quality and preprocessing pipeline."""
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("simulated_logistics_shipments.csv")
print("Rows/columns:", df.shape)
print("Missing values before cleaning:\n", df.isna().sum())
# Remove duplicated shipment identifiers, keeping the first recorded occurrence.
df = df.drop_duplicates(subset="shipment_id", keep="first").copy()
# Convert impossible measurements to missing, then impute numeric fields with median.
df.loc[~df["distance_km"].between(0, 1000), "distance_km"] = float("nan")
df.loc[df["delivery_hours"] < 0, "delivery_hours"] = float("nan")
for col in ["distance_km", "weight_kg", "delivery_hours", "transport_cost_inr"]:
    df[col] = df[col].fillna(df[col].median())
for col in ["traffic_level", "weather"]:
    df[col] = df[col].fillna(df[col].mode().iloc[0])
# Optional robust IQR review: flag rather than automatically delete legitimate long routes.
q1, q3 = df["distance_km"].quantile([0.25, 0.75])
iqr = q3 - q1
outlier_flags = (df["distance_km"] < q1 - 1.5*iqr) | (df["distance_km"] > q3 + 1.5*iqr)
print("Potential distance outliers:", int(outlier_flags.sum()))
# Fit scaling only on numeric features needed for a model; keep IDs/labels untouched.
features = df[["distance_km", "weight_kg", "delivery_hours", "transport_cost_inr"]]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)
print("Cleaned data shape:", df.shape, "Scaled matrix:", scaled.shape)
df.to_csv("cleaned_logistics_shipments.csv", index=False)
