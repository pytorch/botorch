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

Model List GP Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.model_list_gp_regression
    :members:

Multitask GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.multitask
    :members:

Pairwise GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.pairwise_gp
    :members:


Model Components
-------------------------------------------

Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.kernels.downsampling
.. autoclass:: DownsamplingKernel

.. automodule:: botorch.models.kernels.exponential_decay
.. autoclass:: ExponentialDecayKernel

.. automodule:: botorch.models.kernels.linear_truncated_fidelity
.. autoclass:: LinearTruncatedFidelityKernel


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
