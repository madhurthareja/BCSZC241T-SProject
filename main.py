from vidmark import Watermarker

wm = Watermarker(key="secret", strength='medium')
wm.embed("input.mp4", "output.mp4")

