# This is the qasim package, containing the module qasim.qasim in the .so file.
#
# Note that a valid module is one of:
#
# 1. a directory with a modulename/__init__.py file
# 2. a file named modulename.py
# 3. a file named modulename.PLATFORMINFO.so
#
# Since a .so ends up as a module all on its own, we have to include it in a
# parent module (i.e. "package") if we want to distribute other pieces
# alongside it in the same namespace, e.g. qasim_cli.py.
#
name = "qasim"
