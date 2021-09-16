#     The code in this file is modified after
#     https://github.com/dvolgyes/zenodo_get/blob/master/zenodo_get/zget.py

import sys
import os
import json
import signal
import time
import hashlib
import requests
from contextlib import contextmanager

from .utils import download

abort_signal = False
abort_counter = 0
exceptions = False

def ctrl_c(func):

    signal.signal(signal.SIGINT, func)
    return func

@ctrl_c
def handle_ctrl_c(*args, **kwargs):
    global abort_signal
    global abort_counter
    global exceptions

    abort_signal = True
    abort_counter += 1

    if abort_counter >= 2:
        if exceptions:
            raise Exception('\n Immediate abort. There might be unfinished files.')
        else:
            sys.exit(1)


#see https://stackoverflow.com/questions/431684/how-do-i-change-the-working-directory-in-python/24176022#24176022
@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def check_hash(filename, checksum):
    algorithm, value = checksum.split(':')
    if not os.path.exists(filename):
        return value, 'invalid'
    h = hashlib.new(algorithm)
    with open(filename, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            h.update(data)
    digest = h.hexdigest()
    return value, digest


def download_from_zenodo(outdir,
                         doi,
                         cont=False,
                         error=False,
                         **kwargs):
    """
    to suit the requirements of this package.
    :param outdir: Output directory, created if necessary. Default: current directory.
    :param doi: str, Zenodo DOI
    :param cont: True, Do not continue previous download attempt. (Default: continue.)
    :param error: False, Continue with next file if error happens.
    :param kwargs:
        sandbox: bool, Use Zenodo Sandbox URL.
        timeout: int, Connection time-out. Default: 15 [sec].
        pause: float, Seconds to wait before retry attempt, e.g. 0.5
        retry: int, Number of times to Retry on error.
    """
    _wget = kwargs.get('wget', None)
    md5 = kwargs.get('md5', False)
    keep = kwargs.get('keep', False)
    timeout = kwargs.get('timeout', 15)
    sandbox = kwargs.get('sandbox', False)
    pause = kwargs.get('pause', 0.5)
    retry = kwargs.get('retry', 0)

    with cd(outdir):

        url = doi
        if not url.startswith('http'):
            url = 'https://doi.org/' + url
        try:
            r = requests.get(url, timeout=timeout)
        except requests.exceptions.ConnectTimeout:
            raise TimeoutError("Connection timeout.")
        except Exception:
            raise ConnectionError
        if not r.ok:
            raise ValueError(f'DOI {doi} could not be resolved. Try again, or use record ID.')

        recordID = r.url.split('/')[-1]

        if not sandbox:
            url = 'https://zenodo.org/api/records/'
        else:
            url = 'https://sandbox.zenodo.org/api/records/'

        try:
            r = requests.get(url + recordID, timeout=timeout)
        except requests.exceptions.ConnectTimeout:
            raise TimeoutError('Connection timeout during metadata reading.')
        except Exception:
            raise ConnectionError('Connection error during metadata reading.')

        if r.ok:
            js = json.loads(r.text)
            files = js['files']
            total_size = sum(f['size'] for f in files)

            if md5 is not None:
                with open('md5sums.txt', 'wt') as md5file:
                    for f in files:
                        fname = f['key']
                        checksum = f['checksum'].split(':')[-1]
                        md5file.write(f'{checksum}  {fname}\n')

            if _wget is not None:
                if _wget == '-':
                    for f in files:
                        link = f['links']['self']
                        print(link)
                else:
                    with open(_wget, 'wt') as wgetfile:
                        for f in files:
                            fname = f['key']
                            link = 'https://zenodo.org/record/{}/files/{}'.format(
                                recordID, fname
                            )
                            wgetfile.write(link + '\n')
            else:
                print('Title: {}'.format(js['metadata']['title']))
                print('Keywords: ' +
                       (', '.join(js['metadata'].get('keywords', []))))
                print('Publication date: ' + js['metadata']['publication_date'])
                print('DOI: ' + js['metadata']['doi'])
                print('Total size: {:.1f} MB'.format(total_size / 2 ** 20))

                for f in files:
                    if abort_signal:
                        print('Download aborted with CTRL+C.')
                        print('Already successfully downloaded files are kept.')
                        break
                    link = f['links']['self']
                    size = f['size'] / 2 ** 20
                    print()
                    print(f'Link: {link}   size: {size:.1f} MB')
                    fname = f['key']
                    checksum = f['checksum']

                    remote_hash, local_hash = check_hash(fname, checksum)

                    if remote_hash == local_hash and cont:
                        print(f'{fname} is already downloaded correctly.')
                        continue

                    for _ in range(retry + 1):
                        try:
                            filename = download(link)
                        except Exception:
                            print('  Download error.')
                            time.sleep(pause)
                        else:
                            break
                    else:
                        print('  Too many errors.')
                        if not error:
                            raise Exception('Download is aborted. Too  many errors')
                        print('  Download continues with the next file.')
                        continue

                    h1, h2 = check_hash(filename, checksum)
                    if h1 == h2:
                        print(f'Checksum is correct. ({h1})')
                    else:
                        print(f'Checksum is INCORRECT!({h1} got:{h2})')
                        if not keep:
                            print('  File is deleted.')
                            os.remove(filename)
                        else:
                            print('  File is NOT deleted!')
                        if not error:
                            sys.exit(1)
                else:
                    print('All files have been downloaded.')
        else:
            raise Exception('Record could not get accessed.')