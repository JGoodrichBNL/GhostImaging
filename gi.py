# gi.py - CHX Python Ghost Imaging Library
# Author: Justin Goodrich - jgoodrich@bnl.gov
# v0.1

# This is still a work in progress!  Please report bugs or suggest improvements/new functionalities to author.

# Google Doc with information about scans: https://docs.google.com/document/d/1qAZb1ji6fDbVSSwgrpFRU_Enj35iVzhLEoct5f507gs/

import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter

import pyCHX
from pyCHX.chx_packages import *
import pickle as cpk

from collections.abc import Iterable
from collections import defaultdict
import time
    
from tiled.client import from_profile
c = from_profile("chx", "dask")
# c = chx["raw"]
from databroker.queries import TimeRange
results = c.search(TimeRange(since="2021-08"))

# Detector class: stores information about the detector used for the measurements and accesses data from Tiled
    # name: string, human name of the detector
    # did: string, id of the detector in the database
    # pixels: [int, int], number of pixels in the detector 
    # dark: bool, whether or not the detector has dark frames that need to be removed
class Detector:
    def __init__(self, name, did, pixels, dark):
        self.name = name
        self.did = did
        self.pixels = pixels
        self.dark = dark
        
    def get_name(self):
        return self.name
    
    def get_did(self):
        return self.did
    
    def get_pixels(self):
        return self.pixels
    
    def get_dark(self):
        return self.dark    
    
    # load_frames_from_db method: loads number of frames "nframes" starting from ID "sid".  prints controls whether status updates are printed to console every "prints" frames.
    # Note: this function loads the entire frame in; ROI windowing occurs later when data is accessed.
    def load_frames_from_db(self, sid, nframes, scan, prints=False): 
        bucket = np.zeros( [nframes, self.pixels[0], self.pixels[1]] )
        idler = np.zeros( [nframes, self.pixels[0], self.pixels[1]] )
        starttime = time.time()
        print("Accessing database...")
        if scan == False:
            for i in range(nframes):
                if (prints != False):
                    if (i%prints == 0): # 
                        currenttime = time.time()
                        elapsedtime = currenttime - starttime
                        print("Loading frame "+str(i)+"/"+str(nframes-1)+", time elapsed %.1f sec." % elapsedtime)
                if self.dark == True: # for detectors where we need to subtract dark field...
                    bid = sid+3*i
                    iid = sid+3*i+1
                    did = sid+3*i+2
                    bucketf = np.array(results[bid]['primary'].read()[self.did][0])
                    idlerf = np.array(results[iid]['primary'].read()[self.did][0])
                    darkf = np.array(results[did]['primary'].read()[self.did][0])
                    darkroi = darkf[0]
                    bucketroi = bucketf[0]-darkroi
                    idlerroi = idlerf[0]-darkroi          
                elif self.dark == False: # for detectors where we don't need to subtract dark field...
                    bid = sid+2*i
                    iid = sid+2*i+1
                    bucketf = np.array(results[bid]['primary'].read()[self.did][0])
                    idlerf = np.array(results[iid]['primary'].read()[self.did][0])
                    bucketroi = bucketf[0]
                    idlerroi = idlerf[0]
                bucket[i] = bucketroi
                idler[i] = idlerroi
        else: # scan == True, so we have to access the data differently
            data = np.array(results[sid]['primary']['data'].read()['eiger4m_single_image'])
            bucket = data[0:nframes*2:2,0]
            idler = data[1:nframes*2:2,0]
            #for i in range(nframes):
            #    if self.dark == True:
            #        print("Not yet implemented.")
            #    elif self.dark == False:
            #        bid = 2*i
            #        iid = 2*i+1
            #        bucket[i] = data[bid][0]
            #        idler[i] = data[iid][0] 
        currenttime = time.time()
        elapsedtime = currenttime - starttime
        print("Loading complete after %.1f sec." % elapsedtime)        
        print("Bucket/idler shapes are: "+str(bucket.shape))
        return np.array(bucket), np.array(idler)   
    
Detectors = {
    'eiger4m' : Detector("Eiger 4M", "eiger4m_single_image", [2167, 2070], False), # 2167, 2070
    'xray_eye' : Detector("Xray Eye3 CCD", "xray_eye3_image", [2050, 2448], True)
}

# Experiment class: stores information about an experiment
    # name: string, human name/identifier of the experiment
    # desc: string, human description of the experiment
    # detect: Detector object, which was used to collect data
    # date: string, date when the experiment began
    # sid: int, starting ID of the experiment in the database
    # nframes: int, number of frames (bucket and idler is 1 frame) in the experiment to load
    # roistart: [int, int], starting coordinate of the region of interest
    # roisize: [int, int], number of pixels in the region of interest
    # beamsize: int, area of the beam (in micrometers)
class Experiment:
    def __init__(self, name, desc, detect, date, sid, nframes, scan, roistart, roisize, beamsize):
        self.name = name
        self.desc = desc
        self.detect = detect
        self.date = date
        self.sid = sid
        self.nframes = nframes
        self.gframes = nframes # initially this is the number of frames, if frames are later removed for being bad this will adjust
        self.scan = scan
        if (roistart == []): # if no ROI start is defined...
            self.roistart = [0, 0] # ...start at the origin
        else: # ...otherwise...
            self.roistart = roistart # ...save the roi start coordinate
        if (roisize == []): # if no ROI size is defined....
            self.roisize = detect.get_pixels() # ...look at all pixels in the detector
        else: # ...otherwise....
            self.roisize = roisize # ...save the roi size
        self.beamsize = beamsize
        self.bucket = np.array([])
        self.idler = np.array([])
        self.bad = [] # indices of bad frames
        
    def get_name(self):
        return self.name

    def get_desc(self):
        return self.desc

    def get_detect(self):
        return self.detect
    
    def get_date(self):
        return self.date

    def get_sid(self):
        return self.sid

    def get_nframes(self):
        return self.nframes
    
    def get_gframes(self):
        return self.gframes

    def get_roistart(self):
        return self.roistart
    
    def get_roisize(self):
        return self.roisize
    
    def get_beamsize(self):
        if self.beamsize == None:
            return "unspecified"
        else:
            return str(self.beamsize)
    
    def to_string(self):
        return "Experiment '"+self.get_name()+"': \""+self.get_desc()+"\". "+str(self.get_nframes())+" frames taken with "+self.detect.get_name()+" on "+self.get_date()
    
    def set_roisize(self, roisize): 
        print("Changing ROI size to: "+str(roisize))
        self.roisize = roisize
        
    def set_roistart(self, roistart):
        print("Changing ROI start to: "+str(roistart))
        self.roistart = roistart
        
    def set_roi(self, roistart, roiend): # alternative way to define an ROI with a start and end coordinate
        self.roistart = roistart
        self.roisize = roiend-roistart
    
    # load_frames_from_db method: uses the equivalently named method in the Detector class to load data
        # nframes: optional integer, the number of frames to load, if not specified uses the number of frames specified when object was initialized
        # prints: optional integer, print a status message after every "prints" number of frames is loaded
    def load_frames_from_db(self, nframes = False, prints = False):
        if nframes == False:
            nframes = self.get_nframes()
        else:
            self.nframes = nframes
            self.gframes = nframes
            
        print("Loading '"+str(self.get_name())+"', "+str(nframes)+" frames into memory, standby...")    
        self.bucket, self.idler = self.detect.load_frames_from_db(self.get_sid(), nframes, self.scan, prints)
    
    # file_name method method: returns a standard filename to save/load this data
    def get_filename(self):
        return self.get_name() + " " + str(self.get_nframes()) + ".npz"
    
    # load_frames_from_file method: load data from a file, often is faster than loading from Tiled right now
        # filename: optional string, specifies the filename from which to load data 
    def load_frames_from_file(self, filename = False): # load from files
        if filename == False:
            filename = self.get_filename()
        print("Loading frames from file: " + filename + ", standby...")
        starttime = time.time()
        with np.load(filename) as f:
            self.bucket = f['bucket']
            self.idler = f['idler']
        currenttime = time.time()
        elapsedtime = currenttime - starttime
        print("File loading complete after %.1f seconds." % elapsedtime)
        
        
    # save_file_to_frames method: saves data to a file, all frames/pixels are saved
        # filename: optional string, specifies the filename to which to save data
    def save_frames_to_file(self, filename = False):
        if filename == False:
            filename = self.get_filename()
        print("Saving frames to file: " + filename + ", standby...")
        with open(filename, 'wb') as f:
            np.savez(f, bucket=np.array(self.bucket), idler=np.array(self.idler))
        print("File saving complete.")
        
    # get_bucket method: returns bucket data
        # roi: optional bool, if True returns bucket data only over ROI, otherwise returns all data
        # bad: optional bool, if True returns bucket data with bad frames removed, otherwise returns all data
    def get_bucket(self, frames = False, roi = True, bad = True):
        if bad == True and not self.bad == []: # remove bad frames
            bucket = np.delete(self.bucket, self.bad, axis=0)
        else:
            bucket = self.bucket
        if roi == True: # get the ROI from the object
            roistart = self.get_roistart()
            roisize = self.get_roisize()
        else: # otherwise, get the whole frame
            roistart = [0,0]
            roisize = self.detect.get_pixels()
        if frames == False:
            bucket = bucket[:,roistart[0]:roistart[0]+roisize[0],roistart[1]:roistart[1]+roisize[1]]
        else:
            bucket = bucket[frames,roistart[0]:roistart[0]+roisize[0],roistart[1]:roistart[1]+roisize[1]]
        return bucket
    
    # get_idler method: returns bucket data
        # roi: optional bool, if True returns idler data only over ROI, otherwise returns all data
        # bad: optional bool, if True returns idler data with bad frames removed, otherwise returns all data
    def get_idler(self, frames = False, roi = True, bad = True):
        if bad == True and not self.bad == []:
            idler = np.delete(self.idler, self.bad, axis=0)
        else: 
            idler = self.idler
        if roi == True:
            roistart = self.get_roistart()
            roisize = self.get_roisize()
        else:
            roistart = [0,0]
            roisize = self.detect.get_pixels()
        if frames == False:
            idler = idler[:,roistart[0]:roistart[0]+roisize[0],roistart[1]:roistart[1]+roisize[1]]
        else:
            idler = idler[frames,roistart[0]:roistart[0]+roisize[0],roistart[1]:roistart[1]+roisize[1]]
        return idler
    
    # get_data method: returns bucket and idler data
        # roi: optional bool, if True returns data only over ROI, otherwise returns all data
        # bad: optional bool, if True returns data with bad frames removed, otherwise returns all data
    def get_data(self, frames = False, roi = True, bad = True):
        return self.get_bucket(frames = frames, roi = roi, bad = bad), self.get_idler(frames = frames, roi = roi, bad = bad)
    
    # frame_suns method: determines the sum of all pixels in each frame in idler and bucket
        # roi: optional bool, if True it calculates sum over only ROI, otherwise over the entire frame
        # bad: optional bool, if True returns sums with bad frames removed, otherwise returns all data
        # plot: optional bool, if True plots the sums vs frames and their ratios (bucket/idler) 
    def frame_sums(self, roi = True, bad = True, plot = True, ratio = True):
        bucket, idler = self.get_data(roi = roi, bad = bad)
        bsums = bucket.sum(1).sum(1)
        isums = idler.sum(1).sum(1)
        if ratio == True:
            ratio = np.divide(isums, bsums)
        if bad == True:
            frames = self.gframes
        else:
            frames = self.nframes
        if plot == True:
            fig, axs = plt.subplots(2)  
            xlim = self.gframes
            ylim1 = np.median([bsums, isums])*3
            ylim2 = np.median(ratio)*2
            x_list = list(range(0, frames))
            axs[0].set_xlim([0, frames])
            axs[0].set_ylim([0, ylim1])
            axs[0].set_ylabel("Intensity")
            axs[0].plot(x_list, bsums, label="Bucket")
            axs[0].plot(x_list, isums, label="Idler")
            axs[0].legend(loc="upper right")
            axs[1].set_xlim([0, frames])
            axs[1].set_ylim([0, ylim2])
            axs[1].set_xlabel("Frame")
            axs[1].set_ylabel("Idler/Bucket Ratio")
            axs[1].plot(x_list, ratio)        
        return bsums, isums, ratio

    
    # frames_mean method: determines the average intensity of all frames
        # roi: optional bool, if True it calculates sum over only ROI, otherwise over the entire frame
        # bad: optional bool, if True returns mean with bad frames removed, otherwise returns all data  
    def frames_mean(self, roi = True, bad = True):
        sums = self.frame_sums(roi = roi, bad = bad);
        return np.mean(sums[0]), np.mean(sums[1])
    
    # frames_mean method: generates histograms (count vs intensity) of all pixels in each frame
        # frame: int, which frame of the bucket and idler to generate histogram data from
        # roi: optional bool, if True it generates histograms over only the ROI, otherwise over the entire frame
        # bad: optional bool, if True returns histograms of only good frames, otherwise returns all data 
        # bins: optional int, defaults to 500; number of bins in the histograms
    def intensity_histogram(self, frame, roi = True, bad = True, bins = 100, plot = True):
        bucket, idler = self.get_data(frames = frame, roi = roi, bad = bad)
        maxintensity = int(np.max([bucket, idler]))
        print("Max intensity: "+str(maxintensity))
        #bhist = np.array([])
        #ihist = np.array([])
        #for i in range(self.gframes):
        #    np.append(bhist, np.histogram(bucket[i], bins))
        #    np.append(ihist, np.histogram(idler[i], bins))      
        bhist = np.histogram(bucket, bins)
        ihist = np.histogram(idler, bins)
        
        if plot == True:
            max = np.max([bhist[1], ihist[1]])
            fig, axs = plt.subplots(2)  
            x_list = list(range(0, bhist[0].shape[0]))
            axs[0].set_title("Bucket and Idler Histograms")
            axs[0].set_ylim([0, max])
            axs[0].set_xlim([0, maxintensity])
            axs[0].set_ylabel("Count")
            axs[0].plot(bhist[1][:-1], bhist[0])
            axs[1].set_ylim([0, max])
            axs[1].set_xlim([0, maxintensity])
            axs[1].set_xlabel("Intensity")
            axs[1].set_ylabel("Count")
            axs[1].plot(ihist[1][:-1], ihist[0])
        #bhist = np.zeros([maxframes, maxintensity+1])
        #ihist = np.zeros([maxframes, maxintensity+1])
      
        #for i in range(maxframes):
        #    for j in range(bucket.shape[1]):
        #        for k in range(bucket.shape[2]):
        #            bint = int(bucket[i,j,k])
        #            bhist[i][bint] += 1
        #            iint = int(idler[i,j,k])
        #            ihist[i][iint] += 1 
        return bhist, ihist      
        
    # roiarea method: determines the area (number of pixels) in the ROI
    def roiarea(self):
        return self.roisize[0]*self.roisize[1]
 
    # clear_frames method: clears out the bucket, idler, resets bad frames, and resets the good frames to number of total frames
    def clear_frames(self):
        print("Cleared frames in object.")
        self.bucket = np.array([])
        self.idler = np.array([])
        self.bad = []
        self.gframes = self.nframes
    
    # remove_bad_frames method: searches for frames with intensity too far away from the median and returns the indices of these frames
        # sums: optional [int][int][int], if provided uses this data (i.e. if you already calculated it), otherwise calculates it internally
        # scale: optional float, determines how far above (i.e. mean*scale) and below (i.e. mean/scale) the mean is an acceptable range for frame intensity sums.  Should be >= 1 for sensible usage
        # ratio: optional bool, if True it will compare instead to the ratio of idler to bucket sums (i.e. idler/bucket)
        # update: optional bool, if True it will log bad frames into "bad" array and update gframes
        # roi: optional bool, if True it looks for good/bad in the ROI, otherwise over the entire frame
    def remove_bad_frames(self, sums = False, scale = 2.0, ratio = False, update = True, roi = True):
        if sums == False:
            sums = self.frame_sums(roi = roi, bad = False, plot = False)            
            
        if ratio == True:
            ratio = np.divide(sums[1], sums[0])
            ymax = np.median(ratio)*scale
            ymin = np.median(ratio)/scale
            bad1 = list(np.where((ratio > ymax))[0])
            bad2 = list(np.where((ratio < ymin))[0])
            bad = np.concatenate([bad1, bad2])
            bad = np.unique(bad)    
        else:
            bsums = sums[0]
            bmax = np.median(sums[0])*scale
            bmin = np.median(sums[0])/scale
            isums = sums[1]
            imax = np.median(sums[1])*scale
            imin = np.median(sums[1])/scale
            badb1 = list(np.where((bsums > bmax))[0])
            badb2 = list(np.where((bsums < bmin))[0])
            badi1 = list(np.where((isums > imax))[0])
            badi2 = list(np.where((isums < imin))[0])
            bad = np.concatenate([badb1, badb2, badi1, badi2])
            bad = np.unique(bad)
            
        bad = [int(x) for x in bad] # honestly not sure why this is necessary, temporary hack to avoid error
        print("Bad frames found with indices "+str(bad))
        
        if update == True:
            self.bad = bad
            self.gframes = self.nframes - len(bad)
            print("Bad frames saved in object.")
        return bad
    
    # dgi method: applies the DGI to reconstruct the data
        # roi: optional bool, if True it looks for good/bad in the ROI, otherwise over the entire frame   
        # bad: optional bool, if True returns only does DGI over good frames, otherwise all frames    
    def dgi(self, roi = True, bad = True):      
        idler = self.get_idler(roi = roi, bad = bad)
        binsize = idler.shape[0] # should be equal to gframes

        ghost = np.zeros([idler.shape[1], idler.shape[2]])
        bsums, isums, ratio = self.frame_sums(roi, bad, plot = False, ratio = False)
        bsums = bsums
        isums = isums/self.roiarea()
        bsumav = np.mean(bsums)
        isumav = np.mean(isums)

        for i in range(ghost.shape[0]):
            for j in range(ghost.shape[1]):
                sum = 0

                for bin in range(binsize):
                    sum += (bsums[bin] - bsumav*isums[bin]/isumav)*idler[bin][i][j]

                ghost[i,j]=sum/binsize 

        return ghost
    
    # ngi method: applies the NGI to reconstruct the data
    # roi: optional bool, if True it looks for good/bad in the ROI, otherwise over the entire frame   
    # bad: optional bool, if True returns only does NGI over good frames, otherwise all frames    
    def ngi(self, roi = True, bad = True):      
        idler = self.get_idler(roi = roi, bad = bad)
        binsize = idler.shape[0] # should be equal to gframes

        ghost = np.zeros([idler.shape[1], idler.shape[2]])
        bsums, isums, ratio = self.frame_sums(roi, bad, plot = False, ratio = False)
        bsums = bsums
        isums = isums/self.roiarea()
        bsumav = np.mean(bsums)
        isumav = np.mean(isums)

        for i in range(ghost.shape[0]):
            for j in range(ghost.shape[1]):
                sum = 0

                for bin in range(binsize):
                    sum += (bsums[bin]/isums[bin] - bsumav/isumav)*idler[bin][i][j]

                ghost[i,j]=sum/binsize 

        return ghost
    
Experiments = {
    # W G-shaped Wire - trouble loading frames beyond ~600, adjusted obj, should be 1000 though
    'W G-shaped Wire' : Experiment('W G-shaped Wire', 'W G-shaped Wire, 1000 pairs of images with Eiger4m', Detectors['eiger4m'], '8/6/2021', 54733, 600, False, [], [], None),
    'Cardamon Seed 1' : Experiment('Cardamon Seed 1', '1510 frames of cardamon seed taken with Eiger4m', Detectors['eiger4m'], '8/7/2021', 56773, 1000, False, [800,1050], [250,150], None),
    'Cardamon Seed 2' : Experiment('Cardamon Seed 2', 'Continuation of Cardamon Seed 1', Detectors['eiger4m'], '8/9/2021', 59180, 306, False, [], [], None),
    'B Fiber 1' : Experiment('B Fiber 1', 'Static membrane (speckle generator) and sample - B /W fiber', Detectors['eiger4m'], '11/3/2021', 62239, 800, True, [], [], 40),
    'B Fiber 2' : Experiment('B Fiber 2', 'A shorter scan for testing', Detectors['eiger4m'], '11/4/2021', 62240, 50, True, [], [], 40), # double chieck this one
    'B Fiber 3' : Experiment('B Fiber 3', 'Using xray eye, 2000 runs', Detectors['xray_eye'], '4/5/2022', 67666, 2000, False, [], [], 80),
    'B Fiber 4' : Experiment('B Fiber 4', 'Same sample/detector, using larger beam', Detectors['xray_eye'], '4/6/2022', 73690, 3000, False, [], [], 200),
    'B Fiber 5' : Experiment('B Fiber 5', 'Mounted the fiber at a ~45 deg angle.  Feedback loop failed towards end of run (not sure which uid).  Sample_out position clips beam.', Detectors['xray_eye'], '4/7/2022', 82697, 5000, False, [], [], 200), # beam issue
    'B Fiber 6' : Experiment('B Fiber 6', 'Focused beam down, fiber still at angle.  Using eiger4m now.', Detectors['eiger4m'], '4/8/2022', 95141, 1000, False, [1185, 1090], [30, 30], 80)
}
