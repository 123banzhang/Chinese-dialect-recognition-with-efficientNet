import numpy as np
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import argparse
import glob
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import efficientnet_b0 as create_model
import os
import pyaudio
import wave
import wx
from pydub import AudioSegment
#录音
# 每个缓冲区的帧数
CHUNK = 1024
# 采样位数
FORMAT = pyaudio.paInt16
# 单声道
CHANNELS = 1
# 采样频率
RATE = 48000

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)  # ** factor

    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(list(
        map(lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0,
            scale)))  # add list convert
    scale *= (freqbins - 1) / max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]

    return newspec, freqs


""" plot spectrogram"""


def plotstft(audiopath, binsize=2 ** 10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    samples = samples if len(samples.shape) <= 1 else samples[:, channel]
    s = stft(samples, binsize)  # 431 * 513

    # sshow : 431 * 256,
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    ims = ims[0:256, :]  # 0-11khz, ~10s interval
    # print "ims.shape", ims.shape

    image = Image.fromarray(ims)
    image = image.convert('L')
    image.save(name)


def create_spec(input_dir, save_img_dir, audio_type):
    # file = open(input_dir, 'r')
    for iter, line in enumerate(glob.glob(os.path.join(input_dir,
                                                       "*.%s" % audio_type))):  # enumerate(file.readlines()[1:]): # first line of traininData.csv is header (only for trainingData.csv)
        # filepath = line.split(',')[0]
        filename = line.split("/")[-1].split(".")[0]
        # file_split.pop(-1)
        # filename = "_".join(file_split)
        if audio_type == "mp3":
            wavfile = os.path.join(input_dir, filename + '.wav')
            # os.system('ffmpeg -i ' + line +" -ar 8000 tmp.wav" )
            # os.system("ffmpeg -i tmp.wav -ar 8000 tmp.mp3" )
            # os.system("ffmpeg -i tmp.mp3 -ar 8000 -ac 1 "+wavfile )
            os.system('ffmpeg -i ' + line + " " + wavfile)

            # we create only one spectrogram for each speach sample
            # we don't do vocal tract length perturbation (alpha=1.0)
            # also we don't crop 9s part from the speech
            plotstft(wavfile, channel=0, name=os.path.join(save_img_dir, filename + '.png'), alpha=1.0)
            os.remove(wavfile)
            # os.remove("tmp.mp3")
            # os.remove("tmp.wav")
        elif audio_type == "wav":
            plotstft(line, channel=0, name=os.path.join(save_img_dir, filename + '_1.png'), alpha=1.0)
            # print(save_img_dir)
            # print(filename)
        # print ("processed %d files" % (iter + 1));


# 录制声音的相关函数（参数1：录制的路径；参数2：录制的声音秒数）
def record_audio(wave_out_path, record_second):
    # 实例化相关的对象
    p = pyaudio.PyAudio()
    # 打开相关的流，然后传入响应参数
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    # 打开wav文件
    wf = wave.open(wave_out_path, 'wb')
    # 设置相关的声道
    wf.setnchannels(CHANNELS)
    # 设置采样位数8
    wf.setsampwidth(p.get_sample_size((FORMAT)))
    # 设置采样频率
    wf.setframerate(RATE)

    for _ in range(0, int(RATE * record_second / CHUNK)):
        data = stream.read(CHUNK)
        # 写入数据
        wf.writeframes(data)
    # 关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

#GUI
class MyFrame(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, title="大创项目：基于机器学习的方言地域识别系统",
                          pos=(100, 100), size=(600, 400))
        panel = wx.Panel(self)  # 创建画板
        # 创建标题，并设置字体
        title = wx.StaticText(panel, label='基于机器学习的方言地域识别系统', pos=(100, 20))
        font = wx.Font(16, wx.DEFAULT, wx.FONTSTYLE_NORMAL, wx.NORMAL)
        title.SetFont(font)
        # 创建文本和输入框
        self.label_user = wx.StaticText(panel, label="目前支持方言：长沙话，河北话，合肥话，客家话，南昌话，宁夏话，陕西话，上海话，四川话", pos=(50, 70))
        self.label_user = wx.StaticText(panel, label="组员：赵子龙，庞博，张垚杰", pos=(50, 320))
        self.label_user = wx.StaticText(panel, label="请输入录音时长(5秒到10秒):", pos=(50, 100))
        self.text_user = wx.TextCtrl(panel, pos=(230, 100), size=(235, 25), style=wx.TE_LEFT)
        self.picker=wx.FilePickerCtrl(panel, pos=(220, 250), size=(235, 25))
        # self.m_filePicker4 = wx.FileDialog(panel,pos=(250, 100), size=(235, 25))



        # 创建按钮
        self.bt_confirm = wx.Button(panel, label='确定', pos=(480, 100))  # 创建“确定”按钮
        self.bt_confirm.Bind(wx.EVT_BUTTON, self.OnclickSubmit1)
        # 创建文本
        wx.StaticText(panel, label='请点击右侧按钮开始录音', pos=(50, 150))
        wx.StaticText(panel, label='或者选择您的wav格式录音文件', pos=(50, 250))
        self.bt_confirm = wx.Button(panel, label='确定', pos=(480, 250))  # 创建“确定”按钮

        self.bt_confirm.Bind(wx.EVT_BUTTON, self.OnclickSubmit3)
        # 创建按钮
        self.bt_confirm = wx.Button(panel, label='确定', pos=(210, 150))  # 创建“确定”按钮

        self.bt_confirm.Bind(wx.EVT_BUTTON, self.OnclickSubmit2)


    def OnclickSubmit1(self, event): #第一个确定键
        """ 点击确定按钮，执行方法 """
        message = ""
        timeLength = self.text_user.GetValue()     # 获取输入的录音时长
        if  timeLength == "" :    # 判断录音时长是否为空
            message = '时长不能为空'
            wx.MessageBox(message)  # 弹出提示框
        elif int(timeLength) < 5 :  # 录音时长太短
            message = '录音时长太短'
            wx.MessageBox(message)  # 弹出提示框
        elif int(timeLength) > 10 :  # 录音时长太短
            message = '录音时长太长'
            wx.MessageBox(message)  # 弹出提示框

    def m_button1OnButtonClick(self, event):
        # # 键盘控制
        # keyboard = Controller()
        # 获取文件路径
        path = self.picker.GetPath()  # 获取当前选中文件的路径
        if ("wav" in path ) == 0:
            wx.MessageBox("路径错误", "提示",
                          wx.ICON_ERROR)
        else:
            wx.MessageBox('输入成功!', "提示", wx.ICON_INFORMATION)
            event.Skip()

    def OnclickSubmit3(self, event): #第二个确定键
        path = self.picker.GetPath()  # 获取当前选中文件的路径
        # print("path:"+path)
        # print(os.path.dirname(path))
        parser = argparse.ArgumentParser()
        parser.add_argument("--input_dir", type=str, default=os.path.dirname(path),
                            help="")
        parser.add_argument("--save_img_dir", type=str,
                            default=os.path.dirname(path),
                            help="")
        parser.add_argument("--audio_type", type=str, default="wav", help="audio type")

        opt = parser.parse_args()
        print(opt);
        create_spec(input_dir=opt.input_dir, save_img_dir=opt.save_img_dir, audio_type=opt.audio_type)
        filename = path.split("/")[-1].split(".")[0]
        filename=filename + '_1.png'
        img = Image.open(filename).convert("RGB")
        img.save(filename)  # 原地保存

        main(filename)

        os.remove(filename)
    def OnclickSubmit2(self, event): #第二个确定键
        """ 点击确定按钮，执行方法 """

        timeLength = self.text_user.GetValue()
        record_audio(r'C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\录音样本.wav', int(timeLength))

        # ************************************************************************************************
        sound = AudioSegment.from_file(r"C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\录音样本.wav", "wav")
        sound = sound.set_channels(1) #多声道转单声道

        sound.export(r"C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创转换声道后.wav", format="wav")

        # ffmpeg.input(r"转换声道后.wav").output('转换完毕.wav', ar=16000).run()  # 转换采样率
        # frames_per_second = sound.frame_rate
        # print(frames_per_second)
        # channel_count = sound.channels
        # print(channel_count)

        parser = argparse.ArgumentParser()
        parser.add_argument("--input_dir", type=str, default=r'C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创',
                            help="")
        parser.add_argument("--save_img_dir", type=str,
                            default=r"C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创",
                            help="")
        parser.add_argument("--audio_type", type=str, default="wav", help="audio type")

        opt = parser.parse_args()
        print(opt);
        create_spec(input_dir=opt.input_dir, save_img_dir=opt.save_img_dir, audio_type=opt.audio_type)

        img = Image.open(r"C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\录音样本_1.png").convert("RGB")
        img.save(r"C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\录音样本_1.png")  # 原地保存

        main(r"C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\录音样本_1.png")

        # 清理产生的音频
        path1 = r'C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\转换声道后.wav'
        path2 = r'C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\录音样本_1.png"'
        path3 = r'C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\录音样本.wav'
        path4 = r'C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\转换声道后_1.png"'
        os.remove(path1)
        os.remove(path2)
        os.remove(path3)
        os.remove(path4)

        # message = ""
        # number=str(num)
        # if num > 0:
        #     wx.MessageBox("东北话\n识别词数："+number+"\n识别结果：")  # 弹出提示框
        # else:
        #     wx.MessageBox("未识别出结果，请点击确定重新尝试")  # 弹出提示框

def main(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model]),
         transforms.CenterCrop(img_size[num_model]),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path =path #r"C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\录音样本_1.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = r'C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=9).to(device)
    # load model weights
    model_weight_path = r"C:\Users\Pang Bo\Desktop\大三上\大创\2021_Junior\大创\weight\model-29.pth"
    # model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    app = wx.App()                      # 初始化应用
    frame = MyFrame(parent=None, id=-1)  # 实例MyFrame类，并传递参数
    frame.Show()                        # 显示窗口
    for num in range(10, 20):
        app.MainLoop()  # 调用主循环方法



