Noise Modules
=============

The engression mechanism relies on pre-additive stochastic noise layers. These modules are common to both the :doc:`enformer` and :doc:`genformer` architectures. They inject stochastic noise into the continuous representations of the sequence to enable learning the full conditional predictive distribution.

.. autoclass:: genformer.noise.GaussianNoise
   :members:

.. autoclass:: genformer.noise.UniformNoise
   :members:
