import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

# Ensure directory exists
os.makedirs("docs/examples", exist_ok=True)

# 1. Temporal Forecasting Notebook
nb_temporal = new_notebook()

nb_temporal.cells.append(new_markdown_cell("""# Temporal Forecasting with Enformer

This example demonstrates how to use the `Enformer` from the `genformer` package to predict multivariate temporal data. We'll generate a dummy sine wave dataset and create a beautiful probabilistic forecast plot."""))

nb_temporal.cells.append(new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from genformer.models import Enformer

# Style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
"""))

nb_temporal.cells.append(new_code_cell("""# Generate dummy multivariate sine wave data
time_steps = 200
x = np.linspace(0, 50, time_steps)
series1 = (np.sin(x) + np.random.normal(0, 0.1, time_steps)).astype(np.float32)
series2 = (np.cos(x) + np.random.normal(0, 0.1, time_steps)).astype(np.float32)

df = pd.DataFrame({'sin_wave': series1, 'cos_wave': series2})
series = TimeSeries.from_dataframe(df)

# Split data
train, val = series[:-50], series[-50:]

# Plot the dummy data
plt.figure(figsize=(10, 4))
train.plot(label='Training')
val.plot(label='Validation')
plt.title('Dummy Multivariate Time Series Data')
plt.show()
"""))

nb_temporal.cells.append(new_code_cell("""# Initialize Enformer
model = Enformer(
    input_chunk_length=24,
    output_chunk_length=12,
    num_samples_engression=10,
    n_epochs=2, # Keep low for quick demo
    batch_size=16
)

# Train the model (demo purposes)
model.fit(train)
"""))

nb_temporal.cells.append(new_code_cell("""# Predict probabilistic forecasts
# The model natively returns multiple samples forming a predictive distribution
prediction = model.predict(n=50, num_samples=50)

# Plotting the probabilistic forecast
plt.figure(figsize=(12, 6))
series.plot(label='Actual')
prediction.plot(label='Forecast', low_quantile=0.05, high_quantile=0.95)
plt.title('Enformer Probabilistic Forecast')
plt.legend()
plt.show()
"""))

with open("docs/examples/temporal_forecasting_example.ipynb", "w") as f:
    nbf.write(nb_temporal, f)

# 2. Spatiotemporal Forecasting Notebook
nb_spatio = new_notebook()

nb_spatio.cells.append(new_markdown_cell("""# Spatiotemporal Forecasting with Graph-Enformer (GEnformer)

This notebook demonstrates the usage of the `GEnformer` for spatiotemporal data (data with both temporal sequences and spatial topology)."""))

nb_spatio.cells.append(new_code_cell("""import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from genformer.models import GEnformer

plt.style.use('seaborn-v0_8-darkgrid')
"""))

nb_spatio.cells.append(new_code_cell("""# Dummy spatiotemporal data generation: 4 spatial nodes with distinct periodic patterns
time_steps = 150
num_nodes = 4
x = np.linspace(0, 40, time_steps)
data = np.zeros((time_steps, num_nodes), dtype=np.float32)

# Node 0: Sine wave
data[:, 0] = np.sin(x)
# Node 1: Cosine wave (dependent on Node 0)
data[:, 1] = np.cos(x) + 0.3 * data[:, 0]
# Node 2: Faster sine wave (dependent on Node 1)
data[:, 2] = np.sin(1.5 * x) - 0.2 * data[:, 1]
# Node 3: Modulated wave (dependent on Node 0 and Node 2)
data[:, 3] = np.sin(x) * np.cos(2 * x) + 0.15 * data[:, 0] + 0.15 * data[:, 2]

# Add some noise
data += np.random.normal(0, 0.1, (time_steps, num_nodes)).astype(np.float32)

df = pd.DataFrame(data, columns=['Node_0', 'Node_1', 'Node_2', 'Node_3'])
series = TimeSeries.from_dataframe(df)

train, val = series[:-10], series[-10:]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()
for i in range(num_nodes):
    train[f'Node_{i}'].plot(ax=axes[i], label=f'Node {i} (Train)')
    axes[i].legend(loc='upper right')
    axes[i].set_ylabel(f'Node {i}')
fig.suptitle('Spatiotemporal Dummy Data (4 Nodes)', fontsize=14)
plt.tight_layout()
plt.show()
"""))

nb_spatio.cells.append(new_code_cell("""# Create dummy adjacency matrix (edges) for 4 nodes
edges = torch.tensor([
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0]
], dtype=torch.float32)

model = GEnformer(
    input_chunk_length=20,
    output_chunk_length=10,
    edges=edges,
    num_nodes=num_nodes,
    num_samples_engression=5,
    n_epochs=10, # Demo
    batch_size=8,
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1
)

model.fit(train)
"""))

nb_spatio.cells.append(new_code_cell("""from genformer.utils import generate_forecasts

# Generate forecasts using the custom spatial method
# Output shape will be (M, T_out, N, D_gcn) where D_gcn is the latent spatial dimension
# The model expects exactly input_chunk_length as the history window
predictions_tensor = generate_forecasts(
    model=model,
    history=train[-20:], # 20 is the input_chunk_length
    m_samples=30,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# We average over the latent spatial dimension to get the 3 node forecasts
predictions_nodes = predictions_tensor.mean(dim=-1).cpu().numpy() # (M, T_out, N)
"""))

nb_spatio.cells.append(new_code_cell("""# Let's plot the probabilistic spatiotemporal forecast with confidence intervals!
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.flatten()

# Zoom in on the validation period for actuals
actual_zoomed = series[-40:]
actual_time_index = actual_zoomed.time_index if actual_zoomed.has_datetime_index else np.arange(len(series)-40, len(series))

mean_forecast = predictions_nodes.mean(axis=0) # (T_out, N)
q_low = np.quantile(predictions_nodes, 0.05, axis=0) # (T_out, N)
q_high = np.quantile(predictions_nodes, 0.95, axis=0) # (T_out, N)

forecast_len = mean_forecast.shape[0]
if series.has_datetime_index:
    forecast_time_index = series.time_index[len(train):len(train)+forecast_len]
else:
    forecast_time_index = np.arange(len(train), len(train) + forecast_len)

for i in range(num_nodes):
    ax = axes[i]
    color = 'tab:blue'
    forecast_color = 'tab:orange'
    
    # Plot actual
    ax.plot(actual_time_index, actual_zoomed.values()[:, i], label=f'Actual Node {i}', color=color)
    
    # Plot forecast
    ax.plot(forecast_time_index, mean_forecast[:, i], label=f'Forecast Node {i}', color=forecast_color, linestyle='--')
    ax.fill_between(forecast_time_index, q_low[:, i], q_high[:, i], color=forecast_color, alpha=0.3)
    
    ax.legend(loc='upper left')
    ax.set_ylabel(f'Node {i}')

fig.suptitle('GEnformer Spatiotemporal Probabilistic Forecast (4 Nodes)', fontsize=14)
plt.tight_layout()
plt.show()
"""))

with open("docs/examples/spatiotemporal_forecasting_example.ipynb", "w") as f:
    nbf.write(nb_spatio, f)

print("Notebooks created successfully!")
