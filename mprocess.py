## JACOB A TUTMAHER
## STAFF DATA SCIENTIST
## BOOZ ALLEN HAMILTON
##
## 
###############################################################################################
"""ALGORITHM FOR PREPROCESSING DSB3 CT AND MHD DATA - INCLUDING SEGMENTATION. Note: Unless
   specficially intended, users should only change elements in the '__main__' method
   located at the end of this file. The methods should not be changed unless someone
   knows what they are doing.
"""

########################################   IMPORT PK  ######################################### 

import SimpleITK as sitk
import numpy as np
import dicom
import os
import re
import scipy
import logging
import csv
from colorlog import ColoredFormatter
from scipy import ndimage as ndi
from skimage.segmentation import clear_border
from skimage import morphology
from skimage import measure
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage.transform import resize
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing

from skimage import measure, morphology

import multiprocessing
from functools import partial

########################################   BEGIN CODE  ######################################### 

# Get LUNA Files
def get_luna_data(lunadir,annfile):
    # READ IN CSV PATIENT IDS
    csvloc = annfile
    names = []
    with open(csvloc,"rb") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            names = np.append(names, row['seriesuid'])

    # GET UNIQUE NAMES (SOME FILES HAVE MULTIPLE NODULES)
    uniq = np.unique(names)
    cancer = [x+".mhd" for x in uniq]

    # COMPILE CANCER PATIENTS
    subsets = ['subset0','subset1','subset2','subset3','subset4','subset5','subset6','subset7','subset8','subset9']
    subdirs = [lunadir+x for x in subsets]
    lunafiles = []
    for subdir in subdirs:
        temp = [subdir+x for x in os.listdir(subdir) if x in cancer]
        lunafiles = np.append(lunafiles,temp)
    return lunafiles

# Load LUNA MHD files - already in order
def load_mhd_file(path):
    
    image = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(image)
    arr = np.flip(arr,0)
    
    #Set Missing Data to Air
    arr[arr==-3024]=-1024
    arr[arr==-2048]=-1024
    
    return image,arr

# Load DICOM files in order
def load_scans_in_order(path):
    #Order
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    #Get Thickness
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        order_factor = slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        order_factor = slices[0].SliceLocation - slices[1].SliceLocation
    for s in slices:
        s.SliceThickness = slice_thickness
    #Reverse order if scans are upside down
    if order_factor<0:
        slices.reverse()
        
    return slices

#Convert Images to HU
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # JAT Note - This seems sloppy, should probably interpolate these values
    #image[image <= -1024] = -1024
    outside_image = image.min()
    image[image == outside_image] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

# Resample images at different resolution - 1mm x 1mm x 1mm in this case (DICOM)
def resample(image, scan, new_spacing=[1,1,1]):
    # DETERMINE CURRENT PIXEL SPACING
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    # RESIZE
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

# Resample images at different resolution (MHD)
def mhd_resample(image,scan,new_spacing=[1,1,1]):
    
    # DETERMINE CURRENT PIXEL SPACING
    spacing = np.array(scan.GetSpacing())
    spacing = np.roll(spacing,1)
    
    #RESIZE
    resize_factor = spacing/new_spacing
    new_shape = np.round(image.shape*resize_factor)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image,real_resize_factor)
    
    return image, new_spacing

def segment_slice(im):
        ## Original author of this function ArnavJain
    ## https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/run/973430
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    '''
    Step 1: Convert into a binary image.
    '''
    binary = im < -400
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    '''
    Step 8: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    im[get_high_vals] = 0

    return im

# Segment each slice per patient
def segment(patient):
    #THIS CAN TAKE SOME TIME
    return np.asarray([segment_slice(slice) for slice in patient])

# Validate - sometimes segmentation fails and patient data is blank
def validate(myarray):
    uniq = [len(np.unique(myarray[x,:,:])) for x in range(myarray.shape[0])]
    numratio = uniq.count(1)/float(myarray.shape[0])
    logger.debug("Blank Slices: "+str(uniq.count(1)))
    if numratio>0.25:
        return None
    if 1 in uniq:
        return False
    else:
        return True

#Interpolate images with poor segmentation - along z
def interpolate(myarray):
    # JAT NOTE: there's gotta be a better way to do this, 
    #           but its not a major factor in time
    uniq = [len(np.unique(myarray[x,:,:])) for x in range(myarray.shape[0])]
    empty = np.where(np.array(uniq)==1)[0]
    full = np.where(np.array(uniq)!=1)[0]

    #BEGIN INTERPOLATION
    for idx in empty:
        fullidx = np.where(full>idx)[0]
        
        # LINEAR INTERPOLATE ALONG Z
        if len(fullidx)!=0 and idx!=0:
            highidx = min(fullidx)
            low = idx-1
            high = full[highidx]
            width = high-low
            myarray[idx,:,:] = (myarray[low,:,:]+myarray[high,:,:])/float(width)
        
        # CATCH EDGE CASES
        elif idx==0:
            highidx = min(fullidx)
            high = full[highidx]
            myarray[idx,:,:] = myarray[high,:,:]
        else:
            myarray[idx,:,:] = myarray[idx-1,:,:]
    
    return myarray

# Get Label - Class Cancer or Non-cancer
def get_label(patient,labeldict):
    try:
        label = int(labeldict[patient])
    except:
        label = None
    return label

# Average Images into channel data
def average_data(arr,channels=5):
    split = np.array_split(arr,channels,axis=0)
    ave = np.asarray([np.mean(split[i],axis=0) for i in range(channels)])
    return ave.reshape(1,ave.shape[0],ave.shape[1],ave.shape[2])

# Threshold
def threshold(arr,threshold=400):
    arr[arr<-threshold]=0
    arr[arr>threshold]=0
    return arr

def partial_preprocess(input_dict,idx):
    
    # RETRIEVE DATA FROM DICTIONARY
    datafiles = input_dict["datafiles"]
    dicom_dir = input_dict["dicom_dir"]
    luna_dir = input_dict["luna_dir"]
    dicom_dict = input_dict["dicom_dict"]
    channels = input_dict["channels"]

    #GET DATAFILE
    datafile = datafiles[idx]
    logger.info("Begin File Preprocessing: "+datafile+" -- The segmentation might take a minute....")

    #LOAD DEPENDING IF MHD OR DICOM
    if ".mhd" in datafile:
        temp_im,temp_arr = load_mhd_file(datafile)
        name = re.sub(luna_dir,"",datafile)
        name = re.sub(".mhd","",name)
        temp_label = 1 #the MHDs are all cancer
    else:
        temp_im = load_scans_in_order(datafile)
        temp_arr = get_pixels_hu(temp_im)
        name = re.sub(dicom_dir,"",datafile)
        temp_label = get_label(name,dicom_dict)
    
    # SEGMENT LUNGS
    temp_arr = segment(temp_arr)
      
    # CHECK FOR IMPROPER SEGMENTATION AND INTERPOLATE
    check = validate(temp_arr)

    if check==False:
        logger.debug("VALIDATION - File Contains Blank Slice: "+datafile)
        temp_arr = interpolate(temp_arr)
    elif check==None:
        logger.debug("FATAL - File Skipped "+datafile+"\n")
        return np.zeros(temp_arr.shape),0,"null"
    else:
        logger.debug("Image Segmented Correctly: "+datafile)

    #CREATE CHANNEL DATA
    temp_arr = average_data(temp_arr, channels=channels)
    logger.info("Finished File Preprocessing: "+datafile)

    return temp_arr,temp_label,name

def setup_logger():
    # SET LOGGING
    LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
    logging.root.setLevel(LOG_LEVEL)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(LOG_LEVEL)
    stream.setFormatter(formatter)
    logger = logging.getLogger('pythonConfig')
    logger.setLevel(LOG_LEVEL)
    logger.addHandler(stream)
    return logger


# Main Script
if __name__=="__main__":
    """ Does not contain resample, threshold, or zero pad logic """
    
    # USER PARAMETERS
    train_val_ratio = 0.8
    LOG_LEVEL = logging.DEBUG
    DICOM_DIR = "../data/stage1/stage1/"
    DICOM_LABELS = "../data/stage1_labels.csv"
    LUNA_DIR = "../data/LUNA/"
    LUNA_LABELS = "../data/LUNA/CSVFILES/annotations.csv"
    SAVE_DIR = "./preprocess/"
    channels = 5

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    logger = setup_logger()    
    # LOG ENVIRONMENT INFORMATION TO THE USER
    logger.warn("Dicom Directory: "+DICOM_DIR+" -- THIS MIGHT NEED UPDATED ON YOUR MACHINE")
    logger.warn("Dicom Labels: "+DICOM_LABELS+" -- THIS MIGHT NEED UPDATED ON YOUR MACHINE")
    logger.warn("Luna Directory: "+LUNA_DIR+" -- THIS MIGHT NEED UPDATED ON YOUR MACHINE")
    logger.warn("Luna Labels: "+LUNA_LABELS+" -- THIS MIGHT NEED UPDATED ON YOUR MACHINE")
    logger.warn("Save Directory: "+SAVE_DIR+" -- THIS MIGHT NEED UPDATED ON YOUR MACHINE")
    
    # FIND DATA ON LOCAL MACHINE
    lunafiles = get_luna_data(LUNA_DIR,LUNA_LABELS)
    dicomfiles = [DICOM_DIR+x for x in os.listdir(DICOM_DIR) if ".npy" not in x]
    datafiles = list(np.append(dicomfiles,lunafiles))
    logger.info("Number of Data Files Found: "+str(len(datafiles)))

    # LOAD DICOM LABELS
    with open(DICOM_LABELS) as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]:rows[1] for rows in reader}
        infile.close()

    # MULTIPROCESSING
    rows = range(len(datafiles))
    ncores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=ncores)
    logger.info("Parallelizing Across "+str(ncores)+" Cores")

    # SET INPUT DICTIONARY
    d = dict(datafiles=datafiles,dicom_dir=DICOM_DIR,
        luna_dir=LUNA_DIR,dicom_dict=mydict,channels=channels)

    # BEGIN PARALLEL COMPUTATION
    logger.info("Beginning Parallel Computation")
    partial_function = partial(partial_preprocess,d)
    results = pool.map(partial_function,rows)

    # SAFELY CLOSE PROCESSES
    logger.info("Closing multiprocessors")
    pool.close()
    pool.join()

    # PRINT
    logger.info("Postprocessing Parallelized Data")
    filtered_results = [t for t in results if t[2]!="null"]
    test_results = np.array([filtered_results[x][0] for x in range(len(filtered_results))])
    alldata = test_results.reshape(len(filtered_results),channels,512,512)
    labels = np.array([filtered_results[x][1] for x in range(len(filtered_results))])
    names = np.array([filtered_results[x][2] for x in range(len(filtered_results))])
    logger.debug("Data Entries: "+str(alldata.shape)+" Names: "+str(len(names))+" Labels: "+str(len(labels)))

    # ONLY TAKE INNER 3 CHANNELS - THE OUTER 2 ARE USUALLY EDGE DATA ANYWAY
    alldata = alldata[:,1:4,:,:]

    # SAVE ALL DATA
    logger.debug("Saving All Channel Array - This can take some time")
    np.savez_compressed(SAVE_DIR+"all",names=names,data=alldata,labels=labels) 

    #SEPERATE OUT TEST DATA
    testidx = [i for i in range(len(labels)) if labels[i] is None]
    test_arr,test_labels,test_names = alldata[testidx],labels[testidx],names[testidx]
    logger.debug("Saving Test Channel Array - This can take some time")
    np.savez_compressed(SAVE_DIR+"test",data=test_arr,labels=test_labels,names=test_names)

    #AND THE OTHER DATA PLUS RANDOMIZATION
    otheridx = [i for i in range(len(labels)) if labels[i] is not None]
    other_arr,other_labels,other_names = alldata[otheridx],labels[otheridx],names[otheridx]
    p = np.random.permutation(len(other_labels))
    other_arr,other_labels,other_names = other_arr[p],other_labels[p],other_names[p]

    #SPLIT TRAINING AND VALIDATION DATA
    amount = int(train_val_ratio*other_arr.shape[0])
    train_arr,train_labels,train_names = other_arr[0:amount],other_labels[0:amount],other_names[0:amount]
    validate_arr,validate_labels,validate_names = other_arr[amount:],other_labels[amount:],other_names[amount:]
    logger.debug("Saving Train Channel Array - This can take some time")
    np.savez_compressed(SAVE_DIR+"train",data=train_arr,labels=train_labels,names=train_names)
    logger.debug("Saving Validate Channel Array - This can take some time")
    np.savez_compressed(SAVE_DIR+"validate",data=validate_arr,labels=validate_labels,names=validate_names)

    # GET CLASS RATIO FOR SETS
    train_ratio = np.count_nonzero(train_labels)/float(len(train_labels))
    validate_ratio = np.count_nonzero(validate_labels)/float(len(validate_labels))
    test_ratio = np.count_nonzero(test_labels)/float(len(test_labels))

    #LOG SOME FINAL STATISTICS
    logger.info("Preprocessing Complete - Train/Val Ratio "+str(train_val_ratio)+"/"+str(1-train_val_ratio))
    logger.info("-Training Size: "+str(train_arr.shape[0]))
    logger.info("--Class Ratio (Cancer/Non-Cancer): "+str(train_ratio))
    logger.info("-Validation Size: "+str(validate_arr.shape[0]))
    logger.info("--Class Ratio (Cancer/Non-Cancer): "+str(validate_ratio))
    logger.info("-Test Size: "+str(test_arr.shape[0]))
    logger.info("--Class Ratio (Cancer/Non-Cancer): "+str(test_ratio))


