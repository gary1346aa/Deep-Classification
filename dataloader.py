import time
import os
import sys
from six.moves import urllib
import tarfile
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def maybe_download_and_extract(dest_directory, url):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    file_name = 'cifar-10-binary.tar.gz'
    file_path = os.path.join(dest_directory, file_name)
    # if have not downloaded yet
    if not os.path.exists(file_path):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r%.1f%%' % (float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush() # flush the buffer
        print('>> Downloading %s ...' % file_name)
        file_path, _ = urllib.request.urlretrieve(url, file_path, _progress)
        file_size = os.stat(file_path).st_size
        print('\r>> Total %d bytes' % file_size)
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        # Open for reading with gzip compression, then extract all
        tarfile.open(file_path, 'r:gz').extractall(dest_directory)
    print('>> Done')
    