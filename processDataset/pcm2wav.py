import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os
import argparse
import glob

# ## 功能：将pcm文件转成对应wav文件，无压缩
#
# import os
# import wave
#
# def pcm2wav(pcm_path, out_path, channel, sample_rate):
#     with open(pcm_path, 'rb') as pcm_file:
#         pcm_data = pcm_file.read()
#         pcm_file.close()
#     with wave.open(out_path, 'wb') as wav_file:
#         ## 不解之处， 16 // 8， 第4个参数0为何有效
#         wav_file.setparams((channel, 16 // 8, sample_rate, 0, 'NONE', 'NONE'))
#         wav_file.writeframes(pcm_data)
#         wav_file.close()
#
# if __name__ == '__main__':
#     dir = r"C:\Users\Pang Bo\Desktop\test"
#     out_dir = dir + r"\outwav"
#     sample_rate = 48000
#     channel = 1
#     out_path = os.path.join(out_dir, "hebei_train_speaker01_001.wav")
#     pcm2wav(os.path.join(dir, "hebei_train_speaker01_001.pcm"), out_path, channel, sample_rate)
import wave
input='C:/Users/Pang Bo/Desktop/大三下/方言/changsha/dev/speaker'
for i in range(31,36):
    input_dir = input+str(i)+"/short"
    #input_dir = 'C:/Users/Pang Bo/Desktop/大三下/方言/changsha/train/speaker' + str(i)
    audio_type = "pcm"
    # pcm_path = r'C:\Users\Pang Bo\Desktop\test\hebei_train_speaker02_001.pcm'
    for iter, line in enumerate(glob.glob(os.path.join(input_dir, "*.%s" % audio_type))):
        filename = line.split("/")[-1].split(".")[0]
        # print("iter" + str(iter))
        # print("line:" + line)
        print("filename" + filename)
        with open(line, 'rb') as pcmfile:
            pcmdata = pcmfile.read()
        with wave.open(line + '.wav', 'wb') as wavfile:
            wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
            wavfile.writeframes(pcmdata)
        os.remove(line)
