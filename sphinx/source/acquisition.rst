.. role:: hidden
    :class: hidden-section


botorch.acquisition.functional
==============================
.. automodule:: botorch.acquisition.functional
.. currentmodule:: botorch.acquisition.functional


Analytical Acquisition Functions
--------------------------------
.. automodule:: botorch.acquisition.functional.acquisition
   :members:


Batch Acquisition Functions
---------------------------
.. automodule:: botorch.acquisition.functional.batch_acquisition
   :members:


Thompson Sampling Utilities
---------------------------
.. automodule:: botorch.acquisition.functional.thompson_sampling_utils
   :members:


botorch.acquisition.modules
===========================
.. automodule:: botorch.acquisition.modules
.. currentmodule:: botorch.acquisition.modules


Abstract Acquisition Function Module API
----------------------------------------

:hidden:`AcquisitionFunction`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: AcquisitionFunction
   :members:


Acquisition Function Modules
----------------------------

:hidden:`ExpectedImprovement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ExpectedImprovement
   :members:


:hidden:`PosteriorMean`
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PosteriorMean
   :members:


:hidden:`ProbabilityOfImprovement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ProbabilityOfImprovement
   :members:


:hidden:`UpperConfidenceBound`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: UpperConfidenceBound
   :members:


:hidden:`MaxValueEntropySearch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: MaxValueEntropySearch
   :members:


botorch.acquisition.batch_modules
=================================
.. automodule:: botorch.acquisition.batch_modules
.. currentmodule:: botorch.acquisition.batch_modules


Abstract Batch Acquisition Function Module API
----------------------------------------------

:hidden:`BatchAcquisitionFunction`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: BatchAcquisitionFunction
   :members:


Batch Acquisition Function Modules
----------------------------------

:hidden:`qExpectedImprovement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: qExpectedImprovement
   :members:

:hidden:`qNoisyExpectedImprovement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: qNoisyExpectedImprovement
   :members:

:hidden:`qKnowledgeGradient`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: qKnowledgeGradient
   :members:

:hidden:`qKnowledgeGradientNoDiscretization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: qKnowledgeGradientNoDiscretization
   :members:

:hidden:`qProbabilityOfImprovement`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: qProbabilityOfImprovement
   :members:

:hidden:`qSimpleRegret`
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: qSimpleRegret
   :members:

:hidden:`qUpperConfidenceBound`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: qUpperConfidenceBound
   :members:


botorch.acquisition.batch_utils
===============================
.. automodule:: botorch.acquisition.batch_utils
   :members:
