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
import numpy as np
import logging
from colorlog import ColoredFormatter


def setup_logger():
    # SET LOGGING
    LOG_LEVEL = logging.DEBUG
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

if __name__=="__main__":
	# ENVIRONMENT INFO
	LOAD_DIR = "./preprocess_temp/"
	SAVE_DIR = "./preprocess_temp/"
	logger = setup_logger()
	logger.warn("Loading Data From: "+LOAD_DIR)
	logger.warn("Saving Data To: "+SAVE_DIR)

	# LOAD IN ALL.NPZ
	logger.info("Loading all.npz - this could take a minute....")
	alldata_file = np.load("./preprocess_temp/all.npz")
	alldata = alldata_file["data"]
	names = alldata_file["names"]
	labels = alldata_file["labels"]
	train_val_ratio=0.8

	#SEPERATE OUT TEST DATA
    testidx = [i for i in range(len(labels)) if labels[i] is None]
    test_arr,test_labels,test_names = alldata[testidx],labels[testidx],names[testidx]
    logger.debug("Saving Test Channel Array - This can take some time")
    np.savez_compressed(SAVE_DIR+"test",data=test_arr,labels=test_labels,names=test_names)

	#AND THE OTHER DATA PLUS RANDOMIZATION
	logger.info("Breaking Out Training and Validation Data: "+str(train_val_ratio)+"/"+str(1-train_val_ratio))
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