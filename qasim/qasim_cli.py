"""Command-line interface for qasim"""
import sys
from qasim import qasim

qasim.workflow(qasim.get_args(sys.argv[1:]))
