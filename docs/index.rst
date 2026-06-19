:html_theme.sidebar_secondary.remove:

.. raw:: html

   <div class="gf-hero">
     <h1>Genformer</h1>
     <p class="gf-tagline">
       Deep generative Transformers for probabilistic time series and
       spatiotemporal forecasting - uncertainty-aware, robust, and lightweight.
     </p>
     <div class="gf-badges">
       <img src="https://img.shields.io/pypi/v/genformer?color=2e6e63&label=pypi" alt="PyPI">
       <img src="https://img.shields.io/badge/python-3.10%2B-2e6e63" alt="Python">
       <img src="https://img.shields.io/badge/license-MIT-2e6e63" alt="License">
     </div>
     <a class="gf-cta gf-cta-primary" href="usage.html">See it in action</a>
     <a class="gf-cta gf-cta-secondary" href="https://github.com/yuvrajiro/Genformer">GitHub</a>
   </div>

**Genformer** combines the attention mechanism of **Transformers** with the
**engression** paradigm of distributional regression. Rather than point forecasts
or restrictive parametric likelihoods, it injects stochastic noise into the inputs
and optimises a strictly proper **Energy Score**, learning the *full* conditional
predictive distribution and producing diverse, realistic trajectories.

The package ships two models: :doc:`enformer` for temporal data and
:doc:`genformer` for spatiotemporal data on a graph.

Installation
------------

Genformer requires **Python 3.10+** and **PyTorch 2.0+**.

.. tab-set::

   .. tab-item:: PyPI

      .. code-block:: bash

         pip install genformer

   .. tab-item:: From source

      .. code-block:: bash

         git clone https://github.com/yuvrajiro/Genformer.git
         cd Enformer
         pip install -e .

   .. tab-item:: With docs extras

      .. code-block:: bash

         pip install -e ".[docs]"

.. tip::

   For GPU training, install the CUDA build of PyTorch that matches your driver
   *before* installing Genformer - see the
   `PyTorch install matrix <https://pytorch.org/get-started/locally/>`_.

Verify it works:

.. code-block:: python

   from genformer import Enformer, GEnformer
   print("Genformer is ready")

Explore
-------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      The shortest path from a ``TimeSeries`` to a probabilistic forecast, for
      both models.
      +++
      Get going fast

   .. grid-item-card:: Enformer
      :link: enformer
      :link-type: doc

      The temporal model. Pre-additive noise on batch-expanded inputs, trained
      with the Energy Score for probabilistic multivariate forecasting.
      +++
      Temporal forecasting

   .. grid-item-card:: GEnformer
      :link: genformer
      :link-type: doc

      The spatiotemporal model. A graph convolution encodes spatial structure
      before the Transformer, with an optional calibration objective.
      +++
      Spatiotemporal forecasting

   .. grid-item-card:: Usage
      :link: usage
      :link-type: doc

      End-to-end, runnable notebooks for both models - from raw ``TimeSeries``
      to a plotted probabilistic forecast.
      +++
      Worked examples

   .. grid-item-card:: Noise Modules
      :link: noise
      :link-type: doc

      The pre-additive stochastic noise layers shared by both Enformer and GEnformer.
      +++
      Core components

   .. grid-item-card:: Contributors
      :link: contributors
      :link-type: doc

      Meet the contributors and cite Genformer in your research.
      +++
      BibTeX & references

.. toctree::
   :hidden:

   quickstart
   enformer
   genformer
   noise
   usage
   contributors
