from music21 import *

s = converter.parse('Music/Bach/01 Menuet.mid')
s.analyze('key')

