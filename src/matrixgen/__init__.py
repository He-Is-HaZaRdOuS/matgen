"""MatrixGen"""

from scipy.sparse import csr_matrix

from .core import RESIZE_METHODS, resize_matrix
from .utils.features import compute_features
from .utils.io import load_matrix, save_matrix
