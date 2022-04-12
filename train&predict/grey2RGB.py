# from PIL import Image
# import numpy as np
# input='C:/Users/Pang Bo/Desktop/大三下/方言/changsha/dev/speaker'
# image_type = "png"
# for iter, line in enumerate(glob.glob(os.path.join(input_dir, "*.%s" % audio_type))):
#
# im = Image.open(path/to/image, 'r').convert('RGB')
# im = np.stack((im,)*3, axis=-1)
# im = Image.fromarray(im)
# im.save(path/to/save)
import glob
from PIL import Image

for filename in glob.glob(r'D:\Projects\Python_Project\deep-learning-for-image-processing-master\pytorch_classification\Test9_efficientNet\data\dialects\*\*.png'):
    img=Image.open(filename).convert("RGB")
    img.save(filename)#原地保存
