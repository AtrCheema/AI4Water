import os
import requests
import tempfile
import sys, shutil
import urllib.request as ulib
import urllib.parse as urlparse

import bs4


def download_all_http_directory(url, outpath=None, filetypes=None):
    """Download all the files which are of category filetypes at the location of outpath."""
    page = list(urlparse.urlsplit(url))[2].split('/')[-1]
    basic_url = url.split(page)[0]

    r = requests.get(url)
    data = bs4.BeautifulSoup(r.text, "html.parser")
    for l in data.find_all("a"):
        r = requests.get(url + l["href"])
        if l["href"].endswith(".zip"):
            download(basic_url + l["href"])
            print(r.status_code, l["href"], )


def download(url, out=None):
    """High level function, which downloads URL into tmp file in current
    directory and then renames it to filename autodetected from either URL
    or HTTP headers.

    :param url:
    :param out: output filename or directory
    :return:    filename where URL is downloaded to
    """
    # detect of out is a directory
    if out is not None:
        outdir = os.path.dirname(out)
        out_filename = os.path.basename(out)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
    else:
        outdir = os.getcwd()
        out_filename = None

    # get filename for temp file in current directory
    prefix = filename_from_url(url)
    (fd, tmpfile) = tempfile.mkstemp(".tmp", prefix=prefix, dir=".")
    os.close(fd)
    os.unlink(tmpfile)

    # set progress monitoring callback
    def callback_charged(blocks, block_size, total_size):
        # 'closure' to set bar drawing function in callback
        callback_progress(blocks, block_size, total_size, bar_function=bar)

    callback = callback_charged

    # Python 3 can not quote URL as needed
    binurl = list(urlparse.urlsplit(url))
    binurl[2] = urlparse.quote(binurl[2])
    binurl = urlparse.urlunsplit(binurl)

    (tmpfile, headers) = ulib.urlretrieve(binurl, tmpfile, callback)
    filename = filename_from_url(url)

    if out_filename:
        filename = out_filename

    filename = outdir + "/" + filename

    # add numeric ' (x)' suffix if filename already exists
    if os.path.exists(filename):
        filename = filename + '1'
    shutil.move(tmpfile, filename)

    #print headers
    return filename


__current_size = 0

def callback_progress(blocks, block_size, total_size, bar_function):
    """callback function for urlretrieve that is called when connection is
    created and when once for each block

    draws adaptive progress bar in terminal/console

    use sys.stdout.write() instead of "print,", because it allows one more
    symbol at the line end without linefeed on Windows

    :param blocks: number of blocks transferred so far
    :param block_size: in bytes
    :param total_size: in bytes, can be -1 if server doesn't return it
    :param bar_function: another callback function to visualize progress
    """
    global __current_size

    width = 100

    if sys.version_info[:3] == (3, 3, 0):  # regression workaround
        if blocks == 0:  # first call
            __current_size = 0
        else:
            __current_size += block_size
        current_size = __current_size
    else:
        current_size = min(blocks * block_size, total_size)
    progress = bar_function(current_size, total_size, width)
    if progress:
        sys.stdout.write("\r" + progress)


def filename_from_url(url):
    """:return: detected filename as unicode or None"""
    # [ ] test urlparse behavior with unicode url
    fname = os.path.basename(urlparse.urlparse(url).path)
    if len(fname.strip(" \n\t.")) == 0:
        return None
    return fname


def bar(current_size, total_size, width):
    percent = current_size/total_size * 100
    if round(percent % 1, 4) == 0.0:
        print(f"{round(percent)}% of {round(total_size*1e-6, 2)} MB downloaded")
    return