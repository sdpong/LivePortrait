"""
This code file mainly comes from https://github.com/dmlc/gluon-cv/blob/master/gluoncv/utils/filesystem.py
"""
import os
import os.path as osp
import errno
import importlib


def get_model_dir(name, root='~/.insightface'):
    root = os.path.expanduser(root)
    model_dir = osp.join(root, 'models', name)
    return model_dir

def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.

    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def try_import(package, message=None):
    """Try import specified package, with custom message support.

    Parameters
    ----------
    package : str
        The name of the targeting package.
    message : str, default is None
        If not None, this function will raise customized error message when import error is found.


    Returns
    -------
    module if found, raise ImportError otherwise

    """
    try:
        return importlib.import_module(package)
    except ImportError as e:
        if not message:
            raise e
        raise ImportError(message)


def try_import_cv2():
    """Try import cv2 at runtime.

    Returns
    -------
    cv2 module if found. Raise ImportError otherwise

    """
    msg = "cv2 is required, you can install by package manager, e.g. 'apt-get', \
        or `pip install opencv-python --user` (note that this is unofficial PYPI package)."

    return try_import('cv2', msg)


def try_import_mmcv():
    """Try import mmcv at runtime.

    Returns
    -------
    mmcv module if found. Raise ImportError otherwise

    """
    msg = "mmcv is required, you can install by first `pip install Cython --user` \
        and then `pip install mmcv --user` (note that this is unofficial PYPI package)."

    return try_import('mmcv', msg)


def try_import_rarfile():
    """Try import rarfile at runtime.

    Returns
    -------
    rarfile module if found. Raise ImportError otherwise

    """
    msg = "rarfile is required, you can install by first `sudo apt-get install unrar` \
        and then `pip install rarfile --user` (note that this is unofficial PYPI package)."

    return try_import('rarfile', msg)


def import_try_install(package, extern_url=None):
    """Try import the specified package.
    If the package not installed, try use pip to install and import if success.

    Parameters
    ----------
    package : str
        The name of the package trying to import.
    extern_url : str or None, optional
        The external url if package is not hosted on PyPI.
        For example, you can install a package using:
         "pip install git+http://github.com/user/repo/tarball/master/egginfo=xxx".
        In this case, you can pass the url to the extern_url.

    Returns
    -------
    <class 'Module'>
        The imported python module.

    """
    try:
        return importlib.import_module(package)
    except ImportError:
        import subprocess
        import sys

        # trying to install package
        url = package if extern_url is None else extern_url
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', url])

        # trying to load again
        try:
            return importlib.import_module(package)
        except ImportError:
            import site
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.append(user_site)
            return importlib.import_module(package)
    return importlib.import_module(package)


def try_import_dali():
    """Try import NVIDIA DALI at runtime.
    """
    try:
        dali = importlib.import_module('nvidia.dali')
        dali.Pipeline = dali.pipeline.Pipeline
    except ImportError:

        class dali:
            class Pipeline:
                def __init__(self):
                    raise NotImplementedError(
                        "DALI not found, please check if you installed it correctly."
                    )

    return dali
