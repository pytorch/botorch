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

:hidden:`BatchedMultiOutputGPyTorchModel`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: BatchedMultiOutputGPyTorchModel
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

:hidden:`FixedNoiseGP`
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: FixedNoiseGP
  :members:

:hidden:`HeteroskedasticSingleTaskGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: HeteroskedasticSingleTaskGP
  :members:

:hidden:`MultiOutputGP`
~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.models.multi_output_gp_regression
.. autoclass:: MultiOutputGP
  :members:

:hidden:`MultiTaskGP`
~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.models.multitask
.. autoclass:: MultiTaskGP
  :members:

:hidden:`FixedNoiseMultiTaskGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: botorch.models.multitask
.. autoclass:: FixedNoiseMultiTaskGP
   :members:
