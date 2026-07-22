# Intentionally empty.
#
# Its presence at the project root tells pytest to treat this directory as
# the import root, so `from perceptron import Perceptron` resolves correctly
# when running `pytest` from anywhere in the project — without this file,
# pytest would stop path insertion at tests/ (since it has no __init__.py)
# and never see the perceptron/ package next to it.
