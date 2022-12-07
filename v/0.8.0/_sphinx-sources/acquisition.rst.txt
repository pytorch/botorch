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

Cached Cholesky Acquisition Function API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.cached_cholesky
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
	
Multi-Objective Joint Entropy Search Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.joint_entropy_search
    :members:

Multi-Objective Max-value Entropy Search Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.max_value_entropy_search
    :members:

Multi-Objective Monte-Carlo Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.monte_carlo
    :members:
    :exclude-members: MultiObjectiveMCAcquisitionFunction

Multi-Objective Multi-Fidelity Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.multi_fidelity
    :members:
	
Multi-Objective Predictive Entropy Search Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.predictive_entropy_search
    :members:

The One-Shot Knowledge Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.knowledge_gradient
    :members:

Multi-Step Lookahead Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_step_lookahead
    :members:

Max-value Entropy Search Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.max_value_entropy_search
    :members:

Joint Entropy Search Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.joint_entropy_search
    :members:

Predictive Entropy Search Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.predictive_entropy_search
    :members:

Active Learning Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.active_learning
    :members:

Preference Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.preference
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

Risk Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.risk_measures
    :members:

Multi-Output Risk Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.multi_output_risk_measures
    :members:


Utilities
-------------------------------------------

Fixed Feature Acquisition Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.fixed_feature
    :members:

Constructors for Acquisition Function Input Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.input_constructors
    :members:

Penalized Acquisition Function Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.penalized
    :members:

Proximal Acquisition Function Wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.proximal
    :members:

General Utilities for Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.utils
    :members:


Multi-Objective Utilities for Acquisition Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.acquisition.multi_objective.utils
    :members:
