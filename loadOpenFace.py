import sys
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from deepface import DeepFace

try:
    from . SpatialCrossMapLRN_temp import SpatialCrossMapLRN_temp
except:
    try:
        from SpatialCrossMapLRN_temp import SpatialCrossMapLRN_temp
    except:
        SpatialCrossMapLRN_temp = None
import os
import time

import pathlib
containing_dir = str(pathlib.Path(__file__).resolve().parent)


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


#
def Conv2d(in_dim, out_dim, kernel, stride, padding):
    l = torch.nn.Conv2d(in_dim, out_dim, kernel, stride=stride, padding=padding)
    return l

def BatchNorm(dim):
    l = torch.nn.BatchNorm2d(dim)
    return l

def CrossMapLRN(size, alpha, beta, k=1.0, gpuDevice=0):
    if SpatialCrossMapLRN_temp is not None:
        lrn = SpatialCrossMapLRN_temp(size, alpha, beta, k, gpuDevice=gpuDevice)
        n = Lambda( lambda x,lrn=lrn: Variable(lrn.forward(x.data).cuda(gpuDevice)) if x.data.is_cuda else Variable(lrn.forward(x.data)) )
    else:
        n = nn.LocalResponseNorm(size, alpha, beta, k).cuda(gpuDevice)
    return n

def Linear(in_dim, out_dim):
    l = torch.nn.Linear(in_dim, out_dim)
    return l


class Inception(nn.Module):
    def __init__(self, inputSize, kernelSize, kernelStride, outputSize, reduceSize, pool, useBatchNorm, reduceStride=None, padding=True, count=0):
        super(Inception, self).__init__()
        #
        self.counter = count
        self.seq_list = []
        self.outputSize = outputSize

        #
        # 1x1 conv (reduce) -> 3x3 conv
        # 1x1 conv (reduce) -> 5x5 conv
        # ...
        for i in range(len(kernelSize)):
            od = OrderedDict()
            # 1x1 conv
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            # nxn conv
            pad = int(numpy.floor(kernelSize[i] / 2)) if padding else 0
            od['4_conv'] = Conv2d(reduceSize[i], outputSize[i], kernelSize[i], kernelStride[i], pad)
            if useBatchNorm:
                od['5_bn'] = BatchNorm(outputSize[i])
            od['6_relu'] = nn.ReLU()
            #
            self.seq_list.append(nn.Sequential(od))

        ii = len(kernelSize)
        # pool -> 1x1 conv
        od = OrderedDict()
        od['1_pool'] = pool
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od['2_conv'] = Conv2d(inputSize, reduceSize[i], (1,1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['3_bn'] = BatchNorm(reduceSize[i])
            od['4_relu'] = nn.ReLU()
        #
        self.seq_list.append(nn.Sequential(od))
        ii += 1

        # reduce: 1x1 conv (channel-wise pooling)
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od = OrderedDict()
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1,1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            self.seq_list.append(nn.Sequential(od))

        self.seq_list = nn.ModuleList(self.seq_list)


    def forward(self, input):
        x = input

        ys = []
        target_size = None
        depth_dim = 0
        for seq in self.seq_list:
            #print(seq)
            #print(self.outputSize)
            #print('x_size:', x.size())
            y = seq(x)
            y_size = y.size()
            #print('y_size:', y_size)
            ys.append(y)
            #
            if target_size is None:
                target_size = [0] * len(y_size)
            #
            for i in range(len(target_size)):
                target_size[i] = max(target_size[i], y_size[i])
            depth_dim += y_size[1]

        target_size[1] = depth_dim
        #print('target_size:', target_size)

        # padding_values = [
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (3, 3, 4, 4),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (4, 4, 4, 4),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 1, 1),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (2, 2, 2, 2),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 1, 1),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (1, 1, 1, 1),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (1, 1, 1, 1),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (3, 3, 4, 4),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (4, 4, 4, 4),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 1, 1),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (2, 2, 2, 2),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 1, 1),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (1, 1, 1, 1),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (0, 0, 0, 0),  # padding
        #     (1, 1, 1, 1),  # padding
        #     (0, 0, 0, 0)   # padding
        # ]

        padding_values = [
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (3, 3, 4, 4),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (4, 4, 4, 4),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (2, 2, 2, 2),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (3, 3, 4, 4),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (4, 4, 4, 4),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (2, 2, 2, 2),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (0, 0, 0, 0)
        ]

        for i in range(len(ys)):
            y_size = ys[i].size()
            # pad_l = int((target_size[3] - y_size[3]) // 2)
            # pad_t = int((target_size[2] - y_size[2]) // 2)
            # pad_r = target_size[3] - y_size[3] - pad_l
            # pad_b = target_size[2] - y_size[2] - pad_t
            # print("padding")
            # print(pad_l, pad_t, pad_r, pad_b)
            pad_l, pad_t, pad_r, pad_b = padding_values[self.counter + i]
            ys[i] = F.pad(ys[i], (pad_l, pad_r, pad_t, pad_b))
            print(f"Layer {self.counter + i}: Padded size = {ys[i].size()}")
            print(i)

        output = torch.cat(ys, 1)
        return output


class netOpenFace(nn.Module):
    def __init__(self, useCuda, gpuDevice=0):
        super(netOpenFace, self).__init__()

        self.gpuDevice = gpuDevice

        self.layer1 = Conv2d(3, 64, (7,7), (2,2), (3,3))
        self.layer2 = BatchNorm(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        self.layer5 = CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice)
        self.layer6 = Conv2d(64, 64, (1,1), (1,1), (0,0))
        self.layer7 = BatchNorm(64)
        self.layer8 = nn.ReLU()
        self.layer9 = Conv2d(64, 192, (3,3), (1,1), (1,1))
        self.layer10 = BatchNorm(192)
        self.layer11 = nn.ReLU()
        self.layer12 = CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice)
        self.layer13 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        self.layer14 = Inception(192, (3,5), (1,1), (128,32), (96,16,32,64), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True, count=0)
        self.layer15 = Inception(256, (3,5), (1,1), (128,64), (96,32,64,64), nn.LPPool2d(2, (3,3), stride=(3,3)), True, count=4)
        self.layer16 = Inception(320, (3,5), (2,2), (256,64), (128,32,None,None), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True, count=8)
        self.layer17 = Inception(640, (3,5), (1,1), (192,64), (96,32,128,256), nn.LPPool2d(2, (3,3), stride=(3,3)), True, count=11)
        self.layer18 = Inception(640, (3,5), (2,2), (256,128), (160,64,None,None), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True, count=15)
        self.layer19 = Inception(1024, (3,), (1,), (384,), (96,96,256), nn.LPPool2d(2, (3,3), stride=(3,3)), True, count=18)
        self.layer21 = Inception(736, (3,), (1,), (384,), (96,96,256), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True, count=21)
        self.layer22 = nn.AvgPool2d((3,3), stride=(1,1), padding=(0,0))
        self.layer25 = Linear(736, 128)

        #
        self.resize1 = nn.UpsamplingNearest2d(scale_factor=3)
        self.resize2 = nn.AvgPool2d(4)

        #
        # self.eval()

        if useCuda:
            self.cuda(gpuDevice)


    def forward(self, input):
        x = input

        if x.data.is_cuda and self.gpuDevice != 0:
            x = x.cuda(self.gpuDevice)

        #
        if x.size()[-1] == 128:
            x = self.resize2(self.resize1(x))

        x = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        x = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(x)))))
        print('new layer 14')
        x = self.layer14(x)
        print('new layer 15')
        x = self.layer15(x)
        print('new layer 16')
        x = self.layer16(x)
        print('new layer 17')
        x = self.layer17(x)
        print('new layer 18')
        x = self.layer18(x)
        print('new layer 19')
        x = self.layer19(x)
        print('new layer 21')
        x = self.layer21(x)
        print('new layer 22')
        x = self.layer22(x)
        x = x.view((-1, 736))

        x_736 = x

        x = self.layer25(x)
        x_norm = torch.sqrt(torch.sum(x**2, 1) + 1e-6)
        x = torch.div(x, x_norm.view(-1, 1).expand_as(x))
        
        return (x, x_736)


def prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False):
    model = netOpenFace(useCuda, gpuDevice)
    model.load_state_dict(torch.load(os.path.join(containing_dir, 'openface-nocuda.pth')))

    if useMultiGPU:
        model = nn.DataParallel(model)

    return model

#
if __name__ == '__main__':
    #
    useCuda = False
    if useCuda:
        assert torch.cuda.is_available()

    nof = prepareOpenFace(useCuda=useCuda)
    nof = nof.eval()

    # test
    #
    I = numpy.reshape(numpy.array(range(96 * 96), dtype=numpy.float32) * 0.01, (1,96,96))
    I = numpy.concatenate([I, I, I], axis=0)
    I_ = torch.from_numpy(I).unsqueeze(0)

    if useCuda:
        I_ = I_.cuda()

    # print(nof)
    I_ = Variable(I_)
    # print(nof(I_))



    #
    import cv2

    def ReadImage(pathname):
        img = DeepFace.extract_faces(img_path=pathname)[0]["face"]
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
        img = numpy.transpose(img, (2, 0, 1))
        print(img.shape)
        img = img.astype(numpy.float32) / 255.0
        # print(numpy.min(img), numpy.max(img))
        # print(numpy.sum(img[0]), numpy.sum(img[1]), numpy.sum(img[2]))
        I_ = torch.from_numpy(img).unsqueeze(0)
        if useCuda:
            I_ = I_.cuda()
        return I_

    img_paths = ['emmapassport.png']
    imgs = []
    for img_path in img_paths:
        imgs.append(ReadImage(img_path))

    I_ = torch.cat(imgs, 0)
    I_ = Variable(I_, requires_grad=False)

    torch.onnx.export(nof,               # model being run
                      I_,                   # model input (or a tuple for multiple inputs)
                      "openface_pt_native.onnx",            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

    # start = time.time()
    # f, f_736 = nof(I_)
    # print("  + Forward pass took {} seconds.".format(time.time() - start))
    # print(f)
    # for i in range(f_736.size(0) - 1):
    #     for j in range(i + 1, f_736.size(0)):
    #         df = f_736[i] - f_736[j]
    #         print(img_paths[i].split('/')[-1], img_paths[j].split('/')[-1], torch.dot(df, df))

    # in OpenFace's sample code, cosine distance is usually used for f (128d).
    