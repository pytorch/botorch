.. role:: hidden
    :class: hidden-section


botorch.models
========================================================
.. automodule:: botorch.models


Model APIs
-------------------------------------------

Abstract Model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.model
    :members:

GPyTorch Model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.gpytorch
    :members:

Deterministic Model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.deterministic
    :members:


Models
-------------------------------------------

Cost Models (for cost-aware optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.cost
    :members:

GP Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.gp_regression
    :members:

Multi-Fidelity GP Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.gp_regression_fidelity
    :members:

GP Regression Models for Mixed Parameter Spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.gp_regression_mixed
    :members:

Model List GP Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.model_list_gp_regression
    :members:

Multitask GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.multitask
    :members:

Higher Order GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.higher_order_gp
    :members:

Pairwise GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.pairwise_gp
    :members:

Contextual GP Models with Aggregate Rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.contextual
    :members:

Contextual GP Models with Context Rewards
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.contextual_multioutput
    :members:


Model Components
-------------------------------------------

Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.kernels.categorical
.. autoclass:: CategoricalKernel

.. automodule:: botorch.models.kernels.downsampling
.. autoclass:: DownsamplingKernel

.. automodule:: botorch.models.kernels.exponential_decay
.. autoclass:: ExponentialDecayKernel

.. automodule:: botorch.models.kernels.linear_truncated_fidelity
.. autoclass:: LinearTruncatedFidelityKernel

.. automodule:: botorch.models.kernels.contextual_lcea
.. autoclass:: LCEAKernel

.. automodule:: botorch.models.kernels.contextual_sac
.. autoclass:: SACKernel


Transforms
-------------------------------------------

Outcome Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.outcome
    :members:

Input Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.input
    :members:

Transform Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.utils
    :members:


Utilities
-------------------------------------------

Model Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.converter
    :members:

Other Utilties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.utils
    :members:
