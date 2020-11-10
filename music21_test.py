from music21 import *

s = corpus.parse('bach/bwv65.2.xml')
s.analyze('key')