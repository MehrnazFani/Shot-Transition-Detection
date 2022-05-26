import numpy as np
import pandas as pd
import os
import time
import utils
from SBD_for_Mgr import SBD_for_Mgr
"""
Created on Tue Nov 17 18:24:50 2020
@author: mehrnaz
"""
''' This code performs video-shot Transition detection on input video and Parses the input video into (multiple) video-shots
. Input video (x fps, e.g. 60 fps)
. Outputs:
    . video-shots(x/2 fps, e.g. 30 fps)
    . A .csv file that includes video-shots information, i.e., shot_ind, start frame, end frame, start time (sec), end time (sec)

Hierachical temporal segmenting of the video:
     . Main window size for processing the video: The video is devided into Mega-groups of 1minute + 1frame long.
     . Zero-pad the last Mega-group if length of it is not a constant mutliplication of the desired window size (< 60sec + 1 frame).
     . Each mega-group includes 9 groups and has overlap of 1 segment with its previous mega-group
     . Each group includes 10 segments and has overlap of 1 frame with its previous group
     . Each segment includes 21 frames and has ovrlap of 1 frame with its previous segment
Procedure:
    . Thresholding      
'''   
#--- how can I make this code better?
videos_dir = './videos/'
epsilon = 10**(-13) 
# rename videos if .MP4 to .mp4
for file in os.listdir(videos_dir):
    since = time.time()
    if file.endswith('.MP4'):
        os.rename(videos_dir + file, videos_dir + file.split('.MP4')[0]+'.mp4')    
# Process video_files      
for file in os.listdir(videos_dir):
    next = 0   
    if file.endswith('.mp4'): 
        video_name = file  
        frames_dir = videos_dir + 'frames_' + video_name.split('.mp4')[0] + '/'
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)   
            # convert videos into frames with fps/2 frms/sec  and (h/2,w/2) frame resolution     
            print('\\Convert video into frames, with half frame rate and half frame size')
            vid_frm_nums = utils.vid_2_frames_half(videos_dir + video_name, frames_dir)  
        else:
            vid_frm_nums = len(os.listdir(frames_dir))
        print('\\Start Processingthe video:')
        vid_frm_nums = vid_frm_nums - 1 # omit the last frmae just to make sure the all frames are readable 
        indmax = vid_frm_nums -1
        frames_dir = videos_dir + 'frames_' + file.split('.mp4')[0] + '/'       
        window = 60 # 60 secs = 1 minute
        fps = 30 # frame per second 
        Mgr_sz = window * fps + 1  # mega group size = 1801 frms = frame0 -- frame1800
        Mgr_nums = int(vid_frm_nums // (Mgr_sz - 1))
        Mgr_extnd = vid_frm_nums % (Mgr_sz - 1)        
        if Mgr_extnd != 0:
            Mgr_nums += 1
        Mgr_pre_endfrm_num = 20 
        shot_bnd_frms_vid = []
        flash_light_frms_vid = []
        # ************************************************ loop over all Mgrs
        print('Number of Mega-groups:', Mgr_nums)
        time_start_Mgrs = time.time()
        Mgr_pre_endfrms = np.arange(20, (Mgr_nums-1) * (Mgr_sz-1), Mgr_sz - 21)
        for Mgr in range(Mgr_nums):
            shot_bnd_frms, flash_light_frms = SBD_for_Mgr(Mgr, Mgr_sz, Mgr_pre_endfrm_num, vid_frm_nums, frames_dir)
            shot_bnd_frms_vid.extend(shot_bnd_frms)  # extend addes elemets of the list. append adds the whole list       
            flash_light_frms_vid.extend(flash_light_frms) 
            Mgr_pre_endfrm_num +=  Mgr_sz -21
        time_end_Mgrs = time.time() - time_start_Mgrs
        print('all mega groups are analyzed in %d secs' %time_end_Mgrs)
        # here loop of Mgrs ends   
        # ********************************************************************
        """ after video is analyzed, refine all of the shot_boundaries in the video """
        #print('shot_bnd_frms:', shot_bnd_frms_vid)
        #print('flash_light_frms:', flash_light_frms_vid)   
        min_shot_len = 10 # 20 # the minimum possiple shot-lenght 
        vid_frm_max = vid_frm_nums -1
        refined_shot_bnd_frms_vid = []
        
        if shot_bnd_frms_vid and len(shot_bnd_frms_vid) > 1:   #  in a Mega-group     
            """ Omit close boundary frames"""
            F1 = np.array(shot_bnd_frms_vid[:-1])
            FL = np.array(shot_bnd_frms_vid[1:])
            refined_shot_bnd_frms_vid.append(shot_bnd_frms_vid[0])
            diff = FL - F1 <= min_shot_len
            i = 0
            for j in range(0, diff.shape[0]):
                if ~diff[i]:
                    refined_shot_bnd_frms_vid.append(shot_bnd_frms_vid[i + 1])
                i += 1
            # shot_bnd_frms at list inclues [0, ]           
            #bnd_frms = [0] + refined_shot_bnd_frms + [3600]
            f_start_shots = list(np.array(refined_shot_bnd_frms_vid[0:-1]) + 1) # from the first element to one before the end
            f_end_shots = refined_shot_bnd_frms_vid[1:] # from the second element to the end
        if len(shot_bnd_frms_vid) == 1:
            refined_shot_bnd_frms_vid.append(shot_bnd_frms_vid)       
        if len(refined_shot_bnd_frms_vid) == 1:
            f_start_shots = []
            f_end_shots = []
            if refined_shot_bnd_frms_vid[0] >= min_shot_len : #10 or 20
                f_start_shots = [0]
                f_end_shots = [refined_shot_bnd_frms_vid[0]]
            if refined_shot_bnd_frms_vid[0] <= (indmax - min_shot_len): 
                f_start_shots = f_start_shots + [refined_shot_bnd_frms_vid[0] + 1]
                f_end_shots = f_end_shots + [indmax]
            
        if not refined_shot_bnd_frms_vid:
            f_start_shots = [0]
            f_end_shots = [indmax]
        if refined_shot_bnd_frms_vid and (f_start_shots[0] >= min_shot_len):
            f_end_shots = [f_start_shots[0] - 1 ] + f_end_shots 
            f_start_shots = [0] + f_start_shots      
        if refined_shot_bnd_frms_vid and (f_end_shots[-1] <= (indmax - min_shot_len)):  # matrix[-1] = the last element of the matrix
            f_start_shots = f_start_shots + [f_end_shots[-1] + 1]
            f_end_shots = f_end_shots + [indmax]
        
        """ generate a .csv file to save info about all the video-shots and write .mp4 video-shots""" 
        t_start_shots = []
        t_end_shots = []
        for i in range(0, len(f_start_shots)):
            shot_ind = i
            fps = 30 # write videos with 30 frame per second
            f_start_shot = f_start_shots[i] 
            f_end_shot = f_end_shots[i] 
            t_start_shot = round(f_start_shot/fps , 2)
            t_start_shots.append(t_start_shot)
            t_end_shot = round((f_end_shot + 1)/fps, 2)
            t_end_shots.append(t_end_shot)
            shot_name = 'shot' + str(i) + '[' + str(int(t_start_shot // 1))+ '-' + str(int((t_start_shot % 1)*100)) + '--' + str(int(t_end_shot // 1)) + '-' + str(int((t_end_shot % 1)*100)) + ']_' + video_name
            shots_dir = videos_dir + '/videoShots_' + video_name.split('.mp4')[0] + '/'
            if not os.path.exists(shots_dir):
                os.makedirs(shots_dir) 
            utils.video_shot_generate(frames_dir, shots_dir, shot_name, f_start_shot, f_end_shot, fps)
            print('video_shot_%d is generated'%i)
        """ write a .csv file with all shot boundarries for the current video"""
        csvfile_name = 'videoShotsInfo_'+ video_name.split('.mp4')[0] + '.csv'
        data_out = {}
        data_out['shot_ind'] = range(0, len(f_start_shots))
        data_out['frame_start_shot'] = np.array(f_start_shots) # frames start from index 0
        data_out['frame_end_shot'] = np.array(f_end_shots)  # frames start from index 0
        data_out['time_start_shot(sec)'] = t_start_shots
        data_out['time_end_shot(sec)'] = t_end_shots
        df_out = pd.DataFrame(data_out)        
        df_out.to_csv(shots_dir + csvfile_name, index=True)
        print('info about all video-shot is generated in a .csv file')
        
        #---------------------------------------------------------------------
        time_elapsed = time.time() - since
        print('parsing video:'+ video_name +' completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) 
        """DELET FOLDERS INCLUDING FRAMES, WARPED FRAMES and  WARPED MODEL, TO RELEASE THE MEMORY """    
        import shutil
        shutil.rmtree(frames_dir)
        print('Memory released')  
    else:
        next +=1
  
        
