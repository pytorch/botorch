---
id: multi_x_models
title: Multi-X Models
---

In botorch we deal with models that may have multiple outputs, multiple inputs,
and may exploit correlation between between different inputs. We

* "multi-output model": a `Model` (as in the botorch object) with multiple outputs.
* "multi-task": a the logical grouping of inputs/observations (as in the underlying process).
* "multi-task model": A `Model` making use of this logical grouping.

Note the following:
* A multi-task model may or may not be a multi-output model.
* Conversely, a multi-output model may or may not be a multi-output model.
* If a model is both, we refer to it as a multi-task-multi-output model.
