.. role:: hidden
    :class: hidden-section


botorch.models
========================================================
.. automodule:: botorch.models


Model APIs
-------------------------------------------

Base Model API
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

Ensemble Model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.ensemble
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

Variational GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.approximate_gp
    :members:

Fully Bayesian GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.fully_bayesian
    :members:

Fully Bayesian Multitask GP Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.fully_bayesian_multitask
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

Likelihoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.likelihoods.pairwise
    :members:

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

Transform Factory Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.factory
    :members:

Transform Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.transforms.utils
    :members:


Utilities
-------------------------------------------

Dataset Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.utils.parse_training_data
    :members:

Model Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.converter
    :members:

Inducing Point Allocators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.utils.inducing_point_allocators
    :members:
    :private-members: _pivoted_cholesky_init

Other Utilties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: botorch.models.utils.assorted
    :members:
