GEnformer
==============

**Graph-Enformer (GEnformer)** extends :doc:`enformer` to the
**spatiotemporal** setting, where each series is a node in an interconnected
graph (sensors, regions, stations, …). It jointly models temporal dynamics,
spatial interactions, and predictive uncertainty.

How it works
------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Spatial encoding

      A Graph Convolution (GCN) maps each node's features over the graph
      structure to a spatially-aware latent embedding of size ``gcn_out_feat``,
      *before* noise injection and the Transformer backbone.

   .. grid-item-card:: Calibrated uncertainty

      Alongside the Energy Score, an optional calibration loss nudges the
      predictive intervals toward a target coverage level
      (``target_coverage``), weighted by ``lambda_calib``.

The noise injection, ensemble expansion, and Energy Score objective are inherited
from the Enformer recipe.

.. important::

   GEnformer operates and forecasts in the **latent GCN space**. Predictions
   are produced with :func:`genformer.utils.generate_forecasts` (not Darts'
   ``model.predict``) and have shape ``(M, T_out, N, gcn_out_feat)``. Average over
   the last (latent) dimension to recover one forecast per node.

Usage
-----

.. code-block:: python

   import torch
   import numpy as np
   import pandas as pd
   from darts import TimeSeries
   from genformer import GEnformer
   from genformer.utils import generate_forecasts

   # 1. Multivariate series: one column per spatial node
   num_nodes = 3
   df = pd.DataFrame(np.random.randn(150, num_nodes),
                     columns=[f"Node_{i}" for i in range(num_nodes)])
   series = TimeSeries.from_dataframe(df)
   train = series[:-30]

   # 2. Adjacency matrix describing the spatial graph (num_nodes x num_nodes)
   edges = torch.tensor([[0, 1, 1],
                         [1, 0, 1],
                         [1, 1, 0]], dtype=torch.float32)

   # 3. Configure and train
   model = GEnformer(
       input_chunk_length=20,
       output_chunk_length=10,
       edges=edges,
       num_nodes=num_nodes,
       gcn_out_feat=32,
       num_samples_engression=5,
       target_coverage=0.9,        # drives the calibration loss
       lambda_calib=2.0,
       n_epochs=30,
   )
   model.fit(train)

   # 4. Forecast in latent space, then average over the latent dimension
   samples = generate_forecasts(
       model=model,
       history=train[-20:],        # exactly input_chunk_length steps
       m_samples=30,
       device="cuda" if torch.cuda.is_available() else "cpu",
   )                               # (M, T_out, N, gcn_out_feat)

   node_forecasts = samples.mean(dim=-1).cpu().numpy()   # (M, T_out, N)

For a full, runnable walkthrough see :doc:`usage`.

Key hyperparameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Argument
     - Symbol
     - Meaning
   * - ``edges``
     - —
     - Adjacency matrix (``num_nodes`` × ``num_nodes``) of the spatial graph.
   * - ``num_nodes``
     - :math:`N`
     - Number of spatial locations.
   * - ``gcn_out_feat``
     - —
     - Dimensionality of the latent spatial embedding from the GCN.
   * - ``num_samples_engression``
     - :math:`M`
     - Ensemble size used to estimate the Energy Score in training.
   * - ``target_coverage``
     - —
     - Target prediction-interval coverage for the calibration loss.
   * - ``lambda_calib``
     - —
     - Weight of the calibration term in the overall objective.

API reference
-------------

.. autoclass:: genformer.models.GEnformer
   :members:
   :show-inheritance:

.. autofunction:: genformer.utils.generate_forecasts
