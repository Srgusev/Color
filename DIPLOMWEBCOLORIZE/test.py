from PIL import Image
import colorizers
colorizer_eccv16 = colorizers.eccv16().eval()
colorizer_siggraph17 = colorizers.siggraph17().eval()
import numpy as np


out_img = np.array('new.jpg.npy')
out_img.show()