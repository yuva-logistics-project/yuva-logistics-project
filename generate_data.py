"""Generate reproducible simulated logistics shipment data."""
import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
n = 600
zones = rng.choice(["North", "South", "East", "West", "Central"], n)
distance = np.round(rng.uniform(2, 180, n), 1)
weight = np.round(rng.uniform(0.2, 28, n), 2)
traffic = rng.choice(["Low", "Medium", "High"], n, p=[0.3, 0.45, 0.25])
weather = rng.choice(["Clear", "Rain", "Severe"], n, p=[0.68, 0.25, 0.07])
vehicle = rng.choice(["Van", "Truck", "Two-wheeler"], n, p=[0.45, 0.25, 0.30])
priority = rng.choice(["Standard", "Express"], n, p=[0.72, 0.28])
te = pd.Series(traffic).map({"Low": 0, "Medium": 1.2, "High": 3.2}).to_numpy()
we = pd.Series(weather).map({"Clear": 0, "Rain": 1.4, "Severe": 4}).to_numpy()
pe = pd.Series(priority).map({"Standard": 0.8, "Express": -0.5}).to_numpy()
ve = pd.Series(vehicle).map({"Van": 0.4, "Truck": 1, "Two-wheeler": 0.2}).to_numpy()
hours = np.maximum(0.5, 1 + distance/34 + weight*0.045 + te + we + pe + ve + rng.normal(0, 1.15, n))
cost = np.maximum(20, 55 + distance*2.4 + weight*4.2 + pd.Series(vehicle).map({"Van":25,"Truck":60,"Two-wheeler":12}).to_numpy() + (traffic == "High")*35 + rng.normal(0,18,n))
on_time = hours <= (2.5 + distance/45 + (priority == "Express")*1.5)
df = pd.DataFrame({"shipment_id":[f"S{i:04d}" for i in range(1,n+1)], "zone":zones, "distance_km":distance, "weight_kg":weight, "traffic_level":traffic, "weather":weather, "vehicle_type":vehicle, "priority":priority, "delivery_hours":np.round(hours,2), "transport_cost_inr":np.round(cost,2), "on_time":on_time})
df.to_csv("simulated_logistics_shipments.csv", index=False)
print(f"Created {len(df)} simulated shipment records.")
