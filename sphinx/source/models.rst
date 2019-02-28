.. role:: hidden
    :class: hidden-section

botorch.models
===================================

.. automodule:: botorch.models
.. currentmodule:: botorch.models


Abstract Model API
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


:hidden:`MultiOutputGPyTorchModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.gpytorch
.. autoclass:: MultiOutputGPyTorchModel
  :members:


:hidden:`MultiTaskGPyTorchModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.gpytorch
.. autoclass:: MultiTaskGPyTorchModel
  :members:



GPyTorch Regression Models
-----------------------------------

:hidden:`SingleTaskGP`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.gp_regression
.. autoclass:: SingleTaskGP
  :members:
  :exclude-members: forward


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


:hidden:`FidelityAwareSingleTaskGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: botorch.models.fidelity_aware
.. autoclass:: FidelityAwareSingleTaskGP
  :members:
