from vidmark import Watermarker

wm = Watermarker(key="secret", strength="high")
wm.embed(
	"demo.mp4",
	"demo_watermarked.mp4",
)

