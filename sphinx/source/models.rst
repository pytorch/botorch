.. role:: hidden
    :class: hidden-section

botorch.models
===================================

.. automodule:: botorch.models
.. currentmodule:: botorch.models


Abstract API
-----------------------------------

.. currentmodule:: botorch.models.model

:hidden:`Model`
~~~~~~~~~~~~~~~~~~

.. autoclass:: Model
   :members:


:hidden:`GPyTorchModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.gpytorch

.. autoclass:: GPyTorchModel
   :members:


GPyTorch Regression Models
-----------------------------------

:hidden:`SingleTaskGP`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.gp_regression
.. autoclass:: SingleTaskGP
  :members:


:hidden:`HeteroskedasticSingleTaskGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.gp_regression
.. autoclass:: HeteroskedasticSingleTaskGP
  :members:


:hidden:`ConstantNoiseGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.constant_noise
.. autoclass:: ConstantNoiseGP
  :members:


:hidden:`MultiOutputGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.multi_output_gp_regression
.. autoclass:: MultiOutputGP
  :members:
