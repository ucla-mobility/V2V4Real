# Data

I have transformed V2V4real's detection result boxes and ground-truth labels to be in the AB3DMOT's KITTI format and saved then in this github repository.
So the only data we need to download is my V2V4Real's no fusion detection data from both CAVs, which contains the input feature to train and run inference of the DMSTrack model.


## **1. Download my V2V4Real's no fusion detection data from both CAVs**

From this google drive URL (https://drive.google.com/drive/folders/1zOHfwKhHWn5If_xfkpSDzrzjqMxyU2Ni), download no_fusion_keep_all.zip and train_no_fusion_keep_all.zip to the V2V4Real/official_models/ folder and uncompressed them.
Make sure the folder's structure follows:

```
DMSTrack
├── DMSTrack/
├── V2V4Real/
│   ├── official_models/
│   │   ├── no_fusion_keep_all/
│   │   │   ├── npy/
│   │   │   ├── other files
│   │   ├── train_no_fusion_keep_all/
│   │   │   ├── npy/
│   │   │   ├── other files
├── AB3DMOT/
```

## **2. (Optional) Download the V2V4Real's CoBEVT detection data**

If you want to run the tracking inference step using my implementation of V2V4Real's best method: CoBEVT + AB3DMOT, you can download cobevt.zip from the same google drive URL and uncompress it to the same folder as follows:

```
DMSTrack
├── DMSTrack/
├── V2V4Real/
│   ├── official_models/
│   │   ├── cobevt/
│   │   │   ├── npy/
│   │   │   ├── other files
├── AB3DMOT/
```
