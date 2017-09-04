## DataScienceBowl3
Code for all Data Science Bowl 3 written by JAT (In-progress).
Code is currently being updated periodically.

# Data Download
The DICOM images can be found on Kaggles website. You can either download the data there, or via bitTorrent. Options for integrating command line commands with Kaggle competitions can be found here: https://github.com/floydwch/kaggle-cli Simple curl/wgets will only work if configuring them with your browser cookies.

The LUNA images can be found here. There are no easy work arounds on this one. You must set up a profile before getting access to the data via a google doc (seriously). Finding an automated solution for downloading this data can be more hassel than its worths.

DICOM Stage 1 Data Size: 66.88 GB (compressed)
LUNA Data: 66.24 (compressed)

Note: The VoxNet will not work natively with the preprocessing script. It is intended to train on the 3D patient data, interpolated to consistent dimensions. Due to the size of the arrays - each batch must be loaded into memory independently, and the weights saved off after each update.
