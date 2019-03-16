.. role:: hidden
    :class: hidden-section

botorch.models
==============
.. automodule:: botorch.models
.. currentmodule:: botorch.models


Abstract Model API
------------------
.. currentmodule:: botorch.models.model

:hidden:`Model`
~~~~~~~~~~~~~~~
.. autoclass:: Model
   :members:

:hidden:`GPyTorchModel`
~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.models.gpytorch
.. autoclass:: GPyTorchModel
   :members:

:hidden:`MultiOutputGPyTorchModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiOutputGPyTorchModel
  :members:

:hidden:`MultiTaskGPyTorchModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiTaskGPyTorchModel
  :members:


GPyTorch Regression Models
--------------------------
.. currentmodule:: botorch.models.gp_regression

:hidden:`SingleTaskGP`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SingleTaskGP
  :members:
  :exclude-members: forward

:hidden:`HeteroskedasticSingleTaskGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: HeteroskedasticSingleTaskGP
  :members:

:hidden:`ConstantNoiseGP`
~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.models.constant_noise
.. autoclass:: ConstantNoiseGP
  :members:

:hidden:`MultiOutputGP`
~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.models.multi_output_gp_regression
.. autoclass:: MultiOutputGP
  :members:
