import re
import codecs


def convert_unicode(text):
    return codecs.escape_decode(text)[0].decode("utf-8")

