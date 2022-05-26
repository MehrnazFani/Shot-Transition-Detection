"""
utils.py
"""
import cv2
import numpy as np
import math
from tqdm import tqdm
'''START==============utils required for parse_video2shots.py=================='''
# convert video to frames (half frame rate and half frame size in each direction)
def vid_2_frames_half(video_dir, frames_dir):
    image = []
    vidcap = cv2.VideoCapture(video_dir)# video_dir = video_directory + video_name
    frame_number = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_number % 2 !=0 :
        frame_number = frame_number - 1

    #fps_in = vidcap.get(cv2.CAP_PROP_FPS)
    j = 0
    for i in tqdm(range(int(frame_number))):
        _, image = vidcap.read()
        if i%2 == 0: # write frames with the half frame rate 
            #j += 1 # index of frames start frome 1 instead of 0
            image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
            cv2.imwrite(frames_dir + "frame%d" %j + '.jpg', image)  # save frame as JPEG file
            j += 1
            vid_frm_nums = j
        #fps_out = int(fps_in//2)
    return vid_frm_nums   
#-----------------------------------------------------------------------------
def video_shot_generate(frames_dir, shots_dir, shot_name, f_start_shot, f_end_shot, fps):
    img_array = []
    for i in range (f_start_shot, f_end_shot +1 ):
        frm_name = 'frame' + str(i) + '.jpg'
        img = cv2.imread(frames_dir + frm_name)
        img_array.append(img)
    size = img.shape[1], img.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #(*'MP4V')
    # output video dir jumps one level up, compared to frames dir 
    out = cv2.VideoWriter(shots_dir + shot_name, fourcc, fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    cv2.destroyAllWindows()
    out.release()  # releasing the video generated
#-----------------------------------------------------------------------------
def block_3Dcolor_hist(image, blk_num, bin_num):
    """blks_num = number of image partitions
       bin_num = number of histogram bins per color channel
       split image into tile of size MxN"""
    a = math.sqrt(blk_num)
    M = image.shape[0] // int(a)
    N = image.shape[1] // int(a)
    blks = [image[x:x + M, y:y + N, :] for x in range(0, image.shape[0], M) for y in range(0, image.shape[1], N)]
    # color- histogram of each image tile of bins per channel
    hists = np.zeros((blk_num, bin_num ** 3), dtype=np.float64)
    for b in range(blk_num):
        img = cv2.cvtColor(blks[b], cv2.COLOR_BGR2RGB)
        """ Change BGR to RGB, extract a 3D RGB color histogram from the image,
        using bin_num bins per channel, normalize, and update the index 
        cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) """
        hist = cv2.calcHist([img], [0, 1, 2], None, [bin_num, bin_num, bin_num], [0, 256, 0, 256, 0, 256])
        hists[b] = cv2.normalize(hist, hist).flatten()  # normalize in place
    return hists    
#-----------------------------------------------------------------------------
def blk_hist_dist(hist1, hist2):
    return np.mean(np.sum(abs(hist1 - hist2), axis=1), axis=0)    
''' END ===================================================================='''