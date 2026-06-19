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

nb_spatio.cells.append(new_code_cell("""# Dummy data generation: 3 spatial nodes
time_steps = 150
num_nodes = 3
data = np.random.randn(time_steps, num_nodes).astype(np.float32)
# Add some spatial and temporal correlation
data[:, 1] += 0.5 * data[:, 0]
data[:, 2] -= 0.3 * data[:, 1]
for i in range(1, time_steps):
    data[i] += 0.2 * data[i-1]

df = pd.DataFrame(data, columns=['Node_0', 'Node_1', 'Node_2'])
series = TimeSeries.from_dataframe(df)

train, val = series[:-30], series[-30:]

plt.figure(figsize=(10, 4))
train.plot()
plt.title('Spatiotemporal Dummy Data (3 Nodes)')
plt.show()
"""))

nb_spatio.cells.append(new_code_cell("""# Create dummy adjacency matrix (edges)
edges = torch.tensor([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=torch.float32)

model = GEnformer(
    input_chunk_length=20,
    output_chunk_length=10,
    edges=edges,
    num_nodes=num_nodes,
    num_samples_engression=5,
    n_epochs=2, # Demo
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

nb_spatio.cells.append(new_code_cell("""# For simplicity in this demo, let's plot the mean forecast
plt.figure(figsize=(12, 6))
series.plot(label='Actual')

# Mean across samples
mean_forecast = predictions_nodes.mean(axis=0) # (T_out, N)
forecast_len = mean_forecast.shape[0]
time_index = np.arange(len(train), len(train) + forecast_len)

for i in range(num_nodes):
    plt.plot(time_index, mean_forecast[:, i], label=f'Node_{i} Forecast', linestyle='--')

plt.title('GEnformer Spatiotemporal Forecast (Mean)')
plt.legend()
plt.show()
"""))

with open("docs/examples/spatiotemporal_forecasting_example.ipynb", "w") as f:
    nbf.write(nb_spatio, f)

print("Notebooks created successfully!")
