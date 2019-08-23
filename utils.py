"""
Utility functions shared by different runners in this repo
"""


import os
import sys


def initialize_path(*deps):
  root_folder = os.path.dirname(os.path.abspath(sys.argv[0]))
  for dep in deps:
    dep_folder = os.path.join(root_folder, 'deps', dep)
    sys.path.insert(0, dep_folder)


def now():
  return datetime.now().strftime('%H:%M:%S')
