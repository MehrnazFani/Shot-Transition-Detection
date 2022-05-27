#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 17:43:11 2022

@author: mehrnaz
"""
import numpy as np
from numpy import matlib
import cv2
import utils
"""Perfom shot_boundary detection (shot Transition detection) for each mega group of frames"""
def SBD_for_Mgr(Mgr, Mgr_sz, Mgr_pre_endfrm_num, vid_frm_nums, frames_dir):
    print('Analyze Mega-group_', Mgr)
    Mgr_strt_frm = Mgr_pre_endfrm_num -20 
    Mgr_end_frm = Mgr_strt_frm + Mgr_sz -1  #mega group size = 1801 frms = frame0 -- frame1800
    #Mgr_pre_endfrm_num = Mgr_end_frm # to be used in the next loop
    """ Process to be performed on each Mega-group
    # for each video segment compute the distance of marginal frames
    #  indf1[gr_i, sg_j] 9x10: includes start frame ind of (gr_i, sg_j) Where
    #  each Mega-group has 9 groups and each gr has 10 sg and each sg has 21 frms"""
    frm_num = 21  # # of frames in a segment
    sg_num = 10  # # of segments in a group
    gr_num = int(Mgr_sz/200)  # # of groups in a mega-group = 9 in our case
    blk_num = 4#16  # # of blocks of a frame
    bin_num = 3#4  #8 # of histogram bins for each color channel 
    ind_mfrms = np.arange(Mgr_strt_frm, Mgr_end_frm + 1, frm_num-1, dtype=int) #  index of marginal frames of segs in a sequence
    #print('Marginal frames of the mega-group:', ind_mfrms)
    indmax = vid_frm_nums -1
    #indmax = ind_mfrms[-1]
    indf1 = np.reshape(ind_mfrms[0:-1], [gr_num, sg_num]) # 9 x 10
    #indfL = np.reshape(ind_mfrms[1:],[gr_num, sg_num])
    Hist = np.zeros((blk_num, bin_num ** 3,gr_num * sg_num + 1), dtype=np.float64)
    Hist1 = np.zeros((blk_num, bin_num ** 3,gr_num * sg_num), dtype=np.float64)
    Histl = np.zeros((blk_num, bin_num ** 3,gr_num * sg_num), dtype=np.float64)
    #ToDo: may be this loop can be excuted in parallel format
    sg = 0
    for i in ind_mfrms: # this loop calculates block-color histogram of marginal frames in each segment of the Mega-group
        if i < vid_frm_nums :   
            frame = cv2.imread(frames_dir + "/frame" + str(i) + ".jpg")
            hists = utils.block_3Dcolor_hist(frame, blk_num, bin_num)
            Hist[:, :, sg] = hists
            sg += 1
        else: # zero padding
            # if segment is beyond the video frames (i.e., segment is formed by zero-padding the video), keep the histograms zero = initialization 
            Hist[:, :, sg] = 0
            sg += 1                         
    Hist1 = Hist[:, :, 0:-1] # histograms of the first frame of each sg in Mgr
    Histl = Hist[:, :, 1:]# histograms of the last frame of each sg in Mgr
    wloc = 0.5
    wglob = 1
    dloc = utils.blk_hist_dist(Histl, Hist1)
    dglob = np.sum(abs(np.sum(Histl, axis=0) - np.sum(Hist1, axis=0)), axis=0)
    Dsg = wloc * dloc + wglob * dglob
    """threshold for each group : tre_gr = w * mean_gr + (1-w) * ( 1+(ln(mean_Mgr/mean_gr))*sigma_gr )"""
    mean_gr = [np.mean(Dsg[x:x + (sg_num - 1)]) for x in np.arange(0, gr_num * sg_num, sg_num)]
    sigma_gr = [np.std(Dsg[x:x + (sg_num - 1)]) for x in np.arange(0, gr_num * sg_num, sg_num)]
    mean_Mgr = np.mean(mean_gr)
    w = 0.5
    t1 = np.matlib.repmat(np.array(mean_gr), sg_num, 1)
    T1 = np.transpose(t1).flatten()
    t2 = np.matlib.repmat((1 + (np.log(mean_Mgr/(mean_gr))) * np.array(sigma_gr)), sg_num, 1)
    T2 = np.transpose(t2).flatten()
    tre_gr = w * T1 + (1-w) * T2
    cndSgs = Dsg >= tre_gr  # candidate segments that include shot boundary (1 or 0)
    """ compute histogram of all frames of the candidate segments (cndsg) in each group"""
    Hist_gr = np.zeros([blk_num, bin_num ** 3, frm_num, gr_num, sg_num], dtype=np.float64)
    Hdist = np.empty([frm_num - 1, gr_num, sg_num], dtype=np.float64)
    cndSgs_gr = np.reshape(cndSgs, [gr_num, sg_num])
    valMax_Hdist_sg = np.empty([gr_num, sg_num])  # max value of Hdist in a sg
    indMax_Hdist_sg = np.empty([gr_num, sg_num])  # index of the frame in a segment, for which max Hdist is happening
    mean_Hdist_sg = np.empty([gr_num, sg_num])
    median_Hdist_sg = np.empty([gr_num, sg_num])
    sigma_Hdist_sg = np.empty([gr_num, sg_num])
    mean_Hdist_gr = np.empty([gr_num], dtype=np.float64)
    f1 = 0  # index of first frame of a segment
    fL = frm_num - 1  # index of last frame of a segment
    cndShot_bnd_frms = []
    shot_bnd_frms = []
    flash_light_frms = []
    for gr in range(0, gr_num):
        for sg in range(0, sg_num):
            if cndSgs_gr[gr, sg]:
                Hist_gr[:, :, f1, gr, sg] = Hist1[:, :, gr * 10 + sg]
                Hist_gr[:, :, fL, gr, sg] = Histl[:, :, gr * 10 + sg]
                for frm in range(1, frm_num - 1):
                    ind_frm = indf1[gr, sg] + frm
                    if ind_frm <vid_frm_nums : # if frame in range of video
                        frame = cv2.imread(frames_dir + "/frame" + str(ind_frm) + ".jpg")
                        Hist_gr[:, :, frm, gr, sg] = utils.block_3Dcolor_hist(frame, blk_num, bin_num)
                    else: #zero pad if frame out of the vido rang    
                          Hist_gr[:, :, frm, gr, sg] = 0
                """ for each cand segment of the group find the difference
                  hist for all successive frames"""
                for frm in range(0, frm_num -1):
                    Hdist[frm, gr, sg] = utils.blk_hist_dist(Hist_gr[:, :, frm, gr, sg], Hist_gr[:, :, frm + 1, gr, sg])
                """ Round1: find peaks Hdist in each cndsg """
                valMax_Hdist_sg[gr, sg] = np.max(Hdist[:, gr, sg])
                indMax_Hdist_sg[gr, sg] = np.argmax(Hdist[:, gr, sg])
                median_Hdist_sg[gr, sg] = np.median(Hdist[:, gr, sg])      
        mean_Hdist_gr[gr] = np.mean(Hdist[:, gr, cndSgs_gr[gr, :]])
        """ mean_Hist_gr = mean(median(Hdist(cndsgs)))"""
        #mean_Hdist_gr[gr] = np.mean(median_Hdist_sg[gr, cndSgs_gr[gr, :]])
        for sg in range(0, sg_num):
            """ condition 1: Omitting some candidate segments with small distances"""
            if cndSgs_gr[gr, sg]:
                if valMax_Hdist_sg[gr, sg] / (mean_Hdist_gr[gr]) <= 0:
                    # 1.5
                    """ update matrixes"""
                    Hist_gr[:, :, :, gr, sg] = 0
                    Hdist[:, gr, sg] = 0
                    valMax_Hdist_sg[gr, sg] = 0
                    indMax_Hdist_sg[gr, sg] = 0
                    cndSgs_gr[gr, sg] = False       
            """ Round2: find peaks Hdist in each sg & renew parameter values """
            if cndSgs_gr[gr, sg]:
                valMax_Hdist_sg[gr, sg] = np.max(Hdist[:, gr, sg])
                indMax_Hdist_sg[gr, sg] = np.argmax(Hdist[:, gr, sg])
                mean_Hdist_sg[gr, sg] = np.mean(Hdist[:, gr, sg])
                sigma_Hdist_sg[gr, sg] = np.std(Hdist[:, gr, sg])
                median_Hdist_sg[gr, sg] = np.median(Hdist[:, gr, sg])
        mean_Hdist_gr[gr] = np.mean(Hdist[:, gr, cndSgs_gr[gr, :]])
        # mean_Hdist_gr[gr] = np.mean(median_Hdist_sg[gr, cndSgs_gr[gr, :]])
        for sg in range(0, sg_num):
            """ condition 2_1 & condition 2_2: Omitting some candidate segments with small distances"""
            if cndSgs_gr[gr, sg]:
                pk1_sg = valMax_Hdist_sg[gr, sg]
                tmp = np.sort(Hdist[:, gr, sg], axis=0)  # ascending
                # print(tmp)
                tmp_rev = tmp[::-1]  # descending order
                pk2_sg = tmp_rev[1]  # second peak of array
                mean2_Hdist_sg = np.mean(tmp[1:])  # mean of each sg when peak1 is omitted

                cd1 = valMax_Hdist_sg[gr, sg] / (mean_Hdist_gr[gr]) > np.log(mean_Hdist_gr[gr] / (mean_Hdist_sg[gr, sg] * sigma_Hdist_sg[gr, sg] ))  # gives least redundant frames related to flash lights
                #     1.5 * (1 + np.log(mean_Hdist_gr[gr] / (mean_Hdist_sg[gr, sg] * sigma_Hdist_sg[gr, sg])))
                cd2 = ((pk1_sg - pk2_sg) > 0.5)
                cd3 = ((pk1_sg - mean2_Hdist_sg) > np.min([10 *mean2_Hdist_sg, 0.7]))
                cd4 = cd2 and cd3
                cd = cd1 or cd4                        
                if ~ cd:  # no peak is found omit this sg from cand seg list
                    """ update matrixes"""
                    Hist_gr[:, :, :, gr, sg] = 0
                    Hdist[:, gr, sg] = 0
                    valMax_Hdist_sg[gr, sg] = 0
                    indMax_Hdist_sg[gr, sg] = 0
                    cndSgs_gr[gr, sg] = False
                    # shot boundary frames
                    #print(gr, sg)
                else:
                    # print(indf1[gr, sg] + indMax_Hdist_sg[gr, sg])
                    # print('cndShot_bnd_frms=', int(indf1[gr, sg] + indMax_Hdist_sg[gr, sg]))
                    # list of the candidate CT frames
                    cndShot_bnd_frms.append(int(indf1[gr, sg] + indMax_Hdist_sg[gr, sg])) # add ind of the peak as a cnd frame
    #print('cndShot_bnd_frms=', [0 , cndShot_bnd_frms, indmax])
    """ Omitting the flash lights : compare the fram_hist before and after the possible flash-light region:
      if hist is small this is flash light, if it is large shot has changed """
    for f in cndShot_bnd_frms: 
        blk_num = 1
        bin_num = 8
        # If frames in both-sides of cand frame (i.e. f-1 & f+2 )
        # are similar then the cnd frame is flash-light not a boundary frame      
        frame1 = cv2.imread(frames_dir + "/frame" + str(f - 1) + ".jpg")
        hist1 = utils.block_3Dcolor_hist(frame1, blk_num, bin_num)
        if f <= (indmax - 2):
            frame2 = cv2.imread(frames_dir + "/frame" + str(f + 2) + ".jpg")
            hist2 = utils.block_3Dcolor_hist(frame2, blk_num, bin_num)
        elif f <= (indmax -1):
            frame2 = cv2.imread(frames_dir + "/frame" + str(f + 1) + ".jpg")
            hist2 = utils.block_3Dcolor_hist(frame2, blk_num, bin_num)
        else: # zeo pad video
            #frame2 = np.zeros([int(frame1.shape[1] / 2)], int(frame1.shape[0] / 2)])
            #hist2 = block_3Dcolor_hist(frame2, blk_num, bin_num)
            hist2 =  np.zeros((blk_num, bin_num ** 3), dtype=np.float64)
        """ due to distance of frames, motion affects the hist distances more. So, we decrease number of blocks
        to  make the histogram comparison less lensetive to motion"""
        hist_dist = utils.blk_hist_dist(hist1, hist2)
        tre_norule = 0.7
        # print('Dist=', hist_dist)
        if hist_dist > tre_norule:
            shot_bnd_frms.append(f)
        else:
            flash_light_frms.append(f)
    return shot_bnd_frms, flash_light_frms        
