"""Exploratory analysis and plots for simulated logistics data."""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulated_logistics_shipments.csv")
print(df[["distance_km","weight_kg","delivery_hours","transport_cost_inr"]].describe().round(2))
print("Correlation matrix:\n", df[["distance_km","weight_kg","delivery_hours","transport_cost_inr"]].corr().round(2))
print("On-time rate:", round(df["on_time"].mean()*100, 1), "%")
df["delivery_hours"].hist(bins=25)
plt.xlabel("Delivery time (hours)"); plt.ylabel("Shipment count"); plt.title("Distribution of Delivery Times")
plt.tight_layout(); plt.savefig("delivery_time_distribution.png", dpi=160); plt.close()
df.groupby("zone")["delivery_hours"].mean().sort_values().plot(kind="bar")
plt.xlabel("Zone"); plt.ylabel("Average hours"); plt.title("Average Delivery Time by Zone")
plt.tight_layout(); plt.savefig("avg_delivery_by_zone.png", dpi=160); plt.close()
plt.scatter(df["distance_km"], df["delivery_hours"], alpha=.55)
plt.xlabel("Distance (km)"); plt.ylabel("Delivery time (hours)"); plt.title("Distance vs Delivery Time")
plt.tight_layout(); plt.savefig("distance_vs_delivery.png", dpi=160); plt.close()
df.groupby("vehicle_type")["transport_cost_inr"].mean().sort_values().plot(kind="bar")
plt.xlabel("Vehicle type"); plt.ylabel("Average INR"); plt.title("Average Cost by Vehicle Type")
plt.tight_layout(); plt.savefig("cost_by_vehicle.png", dpi=160); plt.close()
