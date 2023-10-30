import numpy as np
import cv2 as cv
import subprocess
from matplotlib import pyplot as plt
from tqdm import tqdm


def video_preprocessing(video_path):
    # 读取文件视频
    cap = cv.VideoCapture(video_path)
    # metdata = skvideo.io.ffprobe("/home/ligaoqi/projects/python_projects/openpose-liyi00-tpami_384_insize_368/video/video_path/2022_3_19/01202.mp4")
    # print(metdata['audio']['tag'])
    # python 读指定路径的文件必须使用反斜杠/或者在路径前面加\r取消路径中的转义字符
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)

    print("待检测视频为{}".format(video_path))
    print("待检测视频高度为{}".format(height))
    print("待视频宽度为{}".format(width))
    print("待视频总帧数为{}".format(count))

    assert '.mp4' in video_path, "Not support other video format except .mp4!"

    video_out_path = video_path.split('.mp4')[0] + '_pre' + '.mp4'
    # 保存视频
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    if height >= width:
        out = cv.VideoWriter(video_out_path, fourcc, fps, (height, width))
    else:
        out = cv.VideoWriter(video_out_path, fourcc, fps, (width, height))
        # out = cv.VideoWriter(video_out_path, fourcc, fps, (height, width))

    # 创建一个 VideoWriter 对象。我们应该指定输出文件名
    # 指定 FourCC 代码。然后传递帧率的数量和帧大小
    # FourCC用于指定视频编解码器的4字节代码,一般采用*mp4v
    for _ in tqdm(range(int(count)), desc='video preprocessing'):
        ret, frame = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            print("video preprocess finished")
            # 此句话会输出,因为读到最后一帧再读一帧就读不到了
            break
        if height >= width:
            frame = np.rot90(frame, 1)
        # else:
        #     frame = np.rot90(frame, -1)

        # cv.imshow('frame', frame)
        out.write(frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()

    return video_out_path


if __name__ == "__main__":
    video_preprocessing('/home/ligaoqi/projects/python_projects/openpose-liyi00-tpami_384_insize_368/video/video_path/2022_3_19/01203.mp4')

