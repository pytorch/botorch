# Contributing to botorch
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

#### Code Style

botorch uses the [black](https://github.com/ambv/black) code formatter to
enforce a common code style across the code base. black is installed easily via
pip using `pip install black`, and run locally by calling
```bash
black .
```
from the repository root. See the [documentation](https://black.readthedocs.io/en/stable/installation_and_usage.html#usage) for more advanced usage.

We feel strongly that having a consistent code style is extremely important, so
Travis will fail on your PR if it does not adhere to the black formatting style.


#### Type Hints
botorch is fully typed using python 3.6+
[type hints](https://www.python.org/dev/peps/pep-0484/).
While we currently do not enforce full consistency of these annotations in
Travis, it is good practice to do so locally. For this, we recommend using
[pyre](https://pyre-check.org/).


#### Testing
To run the unit tests, you can either use `pytest` (if installed):
```bash
pytest -ra
```
or python's `unittest`:
```bash
python -m unittest
```

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.


## License
By contributing to botorch, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
