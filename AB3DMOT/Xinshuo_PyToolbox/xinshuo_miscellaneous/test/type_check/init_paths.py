# Author: Xinshuo Weng
# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import sys, os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

thisdir = os.path.dirname(os.path.abspath(__file__))

python_path = os.path.join(thisdir, '../../')
add_path(python_path)

python_path = os.path.join(thisdir, '../../../')
add_path(python_path)
