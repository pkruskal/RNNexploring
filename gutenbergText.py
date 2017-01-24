__author__ = 'peter'

import requests
import os
from contextlib import closing
import gzip

from _domainModel_text import TEXT_END_MARKERS
from _domainModel_text import TEXT_START_MARKERS
from _domainModel_text import LEGALESE_END_MARKERS
from _domainModel_text import LEGALESE_START_MARKERS

if os.path.exists('.../Data/text'):
    _TEXT_CACHE = '.../Data/text'
elif os.path.exists('../Data/text'):
    _TEXT_CACHE = '../Data/text'
elif os.path.exists('./Data/text'):
    _TEXT_CACHE = './Data/text'
else:
    print('making data path')
    _TEXT_CACHE = './Data/text'
    os.makedirs(os.path.dirname(_TEXT_CACHE))

print('using path ' + _TEXT_CACHE)

def _format_download_uri(etextno):
    """Returns the download location on the Project Gutenberg servers for a
    given text.
    Raises:
        UnknownDownloadUri: If no download location can be found for the text.
    """
    uri_root = r'http://www.gutenberg.lib.md.us'

    if 0 < etextno < 10:
        oldstyle_files = (
            'when11',
            'bill11',
            'jfk11',
            'getty11',
            'const11',
            'liber11',
            'mayfl11',
            'linc211',
            'linc111',
        )
        etextno = int(etextno)
        return '{root}/etext90/{name}.txt'.format(
            root=uri_root,
            name=oldstyle_files[etextno - 1])

    else:
        etextno = str(etextno)
        extensions = ('.txt', '-8.txt', '-0.txt')
        for extension in extensions:
            uri = '{root}/{path}/{etextno}/{etextno}{extension}'.format(
                root=uri_root,
                path='/'.join(etextno[:len(etextno) - 1]),
                etextno=etextno,
                extension=extension)
            response = requests.head(uri)
            if response.ok:
                return uri

    return None


def load_etext(etextno, refresh_cache=False):
    """Returns a unicode representation of the full body of a Project Gutenberg
    text. After making an initial remote call to Project Gutenberg's servers,
    the text is persisted locally.
    """

    cached = _TEXT_CACHE + '/{0}.txt.gz'.format(etextno)

    if refresh_cache:
        remove(cached)
    if not os.path.exists(cached):
        download_uri = _format_download_uri(etextno)
        response = requests.get(download_uri)
        response.encoding = 'utf-8'
        text = response.text
        with closing(gzip.open(cached, 'w')) as cache:
            cache.write(text.encode('utf-8'))
    else:
        with closing(gzip.open(cached, 'r')) as cache:
            text = cache.read().decode('utf-8')
    return text

def strip_headers(text):
    """Remove lines that are part of the Project Gutenberg header or footer.
    Note: this function is a port of the C++ utility by Johannes Krugel. The
    original version of the code can be found at:
    http://www14.in.tum.de/spp1307/src/strip_headers.cpp
    Args:
        text (unicode): The body of the text to clean up.
    Returns:
        unicode: The text with any non-text content removed.
    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out)

