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

Multi-Objective Analytic Acquisition Function API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.acquisition.multi_objective.analytic
.. autoclass:: MultiObjectiveAnalyticAcquisitionFunction
    :members:

Multi-Objective Monte-Carlo Acquisition Function API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.acquisition.multi_objective.monte_carlo
.. autoclass:: MultiObjectiveMCAcquisitionFunction
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

Multi-Objective Analytic Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.analytic
    :members:
    :exclude-members: MultiObjectiveAnalyticAcquisitionFunction

Multi-Objective Monte-Carlo Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.monte_carlo
    :members:
    :exclude-members: MultiObjectiveMCAcquisitionFunction

The One-Shot Knowledge Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.knowledge_gradient
    :members:

Multi-Step Lookahead Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_step_lookahead
    :members:

Entropy-Based Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.max_value_entropy_search
    :members:

Active Learning Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.active_learning
    :members:


Objectives and Cost-Aware Utilities
-------------------------------------------

Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.objective
    :members:

Multi-Objective Objectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.objective
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

Penalized Acquisition Function Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.penalized
    :members:

General Utilities for Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.utils
    :members:
