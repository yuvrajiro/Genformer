Quickstart
==========

From a raw ``TimeSeries`` to a probabilistic forecast in a handful of lines. For
full, runnable notebooks see :doc:`usage`; for the details of each model see
:doc:`enformer` and :doc:`genformer`.

Temporal forecasting with Enformer
---------------------------------------

Enformer follows the standard `Darts <https://unit8co.github.io/darts/>`_
``fit`` / ``predict`` workflow.

.. code-block:: python

   import pandas as pd
   from darts import TimeSeries
   from genformer import Enformer

   # 1. Load your data as a Darts TimeSeries
   series = TimeSeries.from_dataframe(pd.read_csv("your_data.csv"))

   # 2. Configure the model
   model = Enformer(
       input_chunk_length=24,        # look-back window  (p)
       output_chunk_length=12,       # forecast horizon  (q)
       num_samples_engression=10,    # ensemble size     (M)
       noise_dist="gaussian",        # 'gaussian' or 'uniform'
       noise_std=0.1,                # noise scale        (σ)
       n_epochs=30,
   )

   # 3. Train
   model.fit(series)

   # 4. Draw a probabilistic forecast (50 sampled trajectories)
   prediction = model.predict(n=12, num_samples=50)

   # 5. Plot the predictive interval
   prediction.plot(low_quantile=0.05, high_quantile=0.95)

.. admonition:: How the uncertainty appears
   :class: note

   Each call to ``predict`` injects fresh noise into the look-back window, so
   ``num_samples`` independent trajectories are produced. Quantiles over those
   samples form the predictive interval.

Spatiotemporal forecasting with GEnformer
----------------------------------------------

When your series live on a graph (sensors, regions, stations), use
``GEnformer``. It applies a Graph Convolution over the adjacency structure
before the Transformer backbone, and forecasts in **latent GCN space** via
:func:`genformer.utils.generate_forecasts`.

.. code-block:: python

   import torch
   from genformer import GEnformer
   from genformer.utils import generate_forecasts

   # Adjacency matrix (num_nodes x num_nodes) describing the spatial graph
   edges = torch.tensor([[0, 1, 1],
                         [1, 0, 1],
                         [1, 1, 0]], dtype=torch.float32)

   model = GEnformer(
       input_chunk_length=24,
       output_chunk_length=12,
       edges=edges,
       num_nodes=3,
       gcn_out_feat=32,
       num_samples_engression=10,
       target_coverage=0.9,          # drives the calibration loss
       n_epochs=30,
   )
   model.fit(series)

   # Forecast, then average over the latent dimension to get per-node forecasts
   samples = generate_forecasts(model, history=series[-24:], m_samples=30)
   node_forecasts = samples.mean(dim=-1).cpu().numpy()   # (M, T_out, N)

Key hyperparameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Argument
     - Symbol
     - Meaning
   * - ``input_chunk_length``
     - :math:`p`
     - Length of the historical look-back window.
   * - ``output_chunk_length``
     - :math:`q`
     - Forecast horizon.
   * - ``num_samples_engression``
     - :math:`M`
     - In-sample ensemble size used to estimate the Energy Score during training.
   * - ``noise_std``
     - :math:`\sigma`
     - Scale of the injected stochastic noise.
   * - ``noise_dist``
     - —
     - Noise family: ``"gaussian"`` or ``"uniform"``.

Next steps
----------

* Dive into the models: :doc:`enformer` and :doc:`genformer`.
* Work through the runnable notebooks in :doc:`usage`.
