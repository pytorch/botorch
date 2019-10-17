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

Model List GP Regression Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.model_list_gp_regression
    :members:

Multitask GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.multitask
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


Multi-Fidelity Functionality
-------------------------------------------

Multi-Fidelity Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.fidelity.gp_regression_fidelity
    :members:

Multi-Fidelity Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.fidelity_kernels.downsampling
.. autoclass:: DownsamplingKernel

.. automodule:: botorch.models.fidelity_kernels.exponential_decay
.. autoclass:: ExponentialDecayKernel

.. automodule:: botorch.models.fidelity_kernels.linear_truncated_fidelity
.. autoclass:: LinearTruncatedFidelityKernel
