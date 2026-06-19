Enformer
=============

**Enformer** is the temporal model in Genformer. It integrates the
*engression* (distributional regression) principle with a sequence-to-sequence
Transformer to produce **probabilistic** multivariate forecasts.

How it works
------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: 1 · Expand

      The look-back window is replicated into :math:`M` ensemble copies.

   .. grid-item-card:: 2 · Perturb

      Independent pre-additive noise
      :math:`\epsilon^{(m)} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})` is injected
      into each copy.

   .. grid-item-card:: 3 · Score

      The Transformer maps each copy to a trajectory; training minimises the
      strictly proper **Energy Score**.

Because the Energy Score is a strictly proper scoring rule, minimising it drives
the sampled trajectories toward the true conditional predictive distribution —
no parametric likelihood assumption required.

Usage
-----

Enformer follows the standard `Darts <https://unit8co.github.io/darts/>`_
``fit`` / ``predict`` workflow over a ``TimeSeries``:

.. code-block:: python

   import pandas as pd
   from darts import TimeSeries
   from genformer import Enformer

   # 1. Load data as a Darts TimeSeries
   series = TimeSeries.from_dataframe(pd.read_csv("your_data.csv"))
   train, val = series[:-50], series[-50:]

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
   model.fit(train)

   # 4. Draw a probabilistic forecast (50 sampled trajectories)
   prediction = model.predict(n=12, num_samples=50)

   # 5. Plot the predictive interval
   prediction.plot(low_quantile=0.05, high_quantile=0.95)

.. admonition:: Where the uncertainty comes from
   :class: note

   Every ``predict`` call injects fresh noise into the look-back window, so
   ``num_samples`` independent trajectories are produced. Quantiles over those
   samples form the predictive interval.

For a full, runnable walkthrough see :doc:`usage`.

Key hyperparameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

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
     - In-sample ensemble size used to estimate the Energy Score in training.
   * - ``noise_std``
     - :math:`\sigma`
     - Scale of the injected stochastic noise.
   * - ``noise_dist``
     - —
     - Noise family: ``"gaussian"`` or ``"uniform"``.

API reference
-------------

.. autoclass:: genformer.models.Enformer
   :members:
   :show-inheritance:

Noise modules
~~~~~~~~~~~~~~

The engression mechanism is driven by pre-additive stochastic noise layers.

.. autoclass:: genformer.noise.GaussianNoise
   :members:

.. autoclass:: genformer.noise.UniformNoise
   :members:
