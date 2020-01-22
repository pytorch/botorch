.. role:: hidden
    :class: hidden-section


botorch.acquisition
========================================================
.. automodule:: botorch.acquisition


Acquisition Function APIs
-------------------------------------------

Abstract Acquisition Function APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.acquisition
    :members:

Analytic Acquisition Function API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.acquisition.analytic
.. autoclass:: AnalyticAcquisitionFunction
    :members:

Monte-Carlo Acquisition Function API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.acquisition.monte_carlo
.. autoclass:: MCAcquisitionFunction
    :members:


Acquisition Functions
-------------------------------------------

Analytic Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.analytic
    :members:
    :exclude-members: AnalyticAcquisitionFunction

Monte-Carlo Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.monte_carlo
    :members:
    :exclude-members: MCAcquisitionFunction

The One-Shot Knowledge Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.knowledge_gradient
    :members:

Entropy-Based Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.max_value_entropy_search
    :members:

Multi-Step Look-Ahead Acquisition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_step_lookahead
    :members:


Objectives and Cost-Aware Utilities
-------------------------------------------

Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.objective
    :members:

Cost-Aware Utility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.cost_aware
    :members:


Utilities
-------------------------------------------

Fixed Feature Acquisition Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.fixed_feature
    :members:

General Utilities for Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.utils
    :members:
