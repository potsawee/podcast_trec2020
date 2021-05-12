CUED_SPEECH @ TREC2020 Podcast Summarisation Track
=====================================================
 
Overview
--------------------------------------
- Report: [**CUED_SPEECH AT TREC 2020 PODCAST SUMMARISATION TRACK**](https://arxiv.org/abs/2012.02535)
- It is a two-stage approach as the BART model can only take up to 1024 input words:
	- Stage1: Truncation or Filtering (by Hierarchical Model)
	- Stage2: Running BART
- Scripts have absolute paths (to model/data), typically variables in CAPITAL => change them to your own paths.
- Feel free to email <pm574@cam.ac.uk> if you think this repository might be useful for your work, but it's not working. 	

Requirements
--------------------------------------
- python 3.7
- torch 1.2.0
- transformers (HuggingFace) 2.11.0

(I expect it to work with newer versions too, but not guranteed & not tested)

Data Preparation
--------------------------------------
**Spotify Podcast: Download & Pre-processing**

- Download link: <https://podcastsdataset.byspotify.com/>
- ```data/processor.py```: split the data into chunks such that each chuck contains 10k instance, e.g. id0-id9999 in podcast_set0 (we could use your own data processing pipeline!!). Note that to use our trained weights, we use BART tokenizer to process the data.
- ```data/processor_testset.py```: for pre-processing test data
- ```data/loader.py``` contains functions (mostly hard coded) for data loading, batching, etc.

Train & Fine-tune BART 
--------------------------------------
*Currently, all the configurations must be set in the training script!!

**Standard Fine-tuning**:

    python train_bartvanilla.py

**L_rl training**:

	python train_bartvanilla_rl.py

Decode (Inference) BART
--------------------------------------
*Again, all configutations must be set in the script (note that the current settings are the one used in TREC2020)

**decoding**:

    python decode_testset.py start_id end_id
    
- The test set consists of 1,027 samples. See ```data/processor_testset.py``` about how to prepare the test data. If used in a single machine, you can unable ID randomization.
- To use sentence filtering at test time, you need to decode the test data using the hierarchical model first as it is a two-stage process. Refer to Decode Hierarchical Model section.

**ensemble decoding** ("token-level combination / product-of-expectations"):

    python ensemble_decode_testset.py start_id end_id
    
 
    
Train Hierarchical Model
--------------------------------------
For performing sentence filtering, i.e. content selection


    python train_hiermodel.py

Configurations are set inside this script.

Decode Hierarchical Model section 
--------------------------------------
This is the first stage before BART decoding unless you do truncation at test time:

	python hier_filtering_testset.py start_id end_id

If you want to train BART with filtered data:

	python hier_filtering_trainset.py start_id end_id

Trained Weights
-----------------------------------------
OneDrive link (~7.5GB): 
1. If you are in Cambridge (most likely not) [Cambridge Access Link](https://universityofcambridgecloud-my.sharepoint.com/:u:/g/personal/pm574_cam_ac_uk/EWzdmdaYVJdDpzxpvXX6KRUBnQvbmZBbhpSQQ2F-vXJlqg)
2. For external people, please email me <pm574@cam.ac.uk> and I will add your email to the sharing list. 
Sorry for the inconvenice, the file is too large that I have to host it on the Cambridge OneDrive, but it doesn't allow me to share with link.  

**BART**

- cued\_speech\_bart\_baseline.pt (run\_id: cued\_speechUniv3)
- cued\_speech\_bart\_filtered.pt (run\_id: cued\_speechUniv4)
- cued\_speech\_ensemble3\_x.pt (run\_id: cued\_speechUniv2)
	- cued\_speech\_ensemble3_1.pt
	- cued\_speech\_ensemble3_2.pt
	- cued\_speech\_ensemble3_3.pt

**Hierarchical Model**

- HIERMODEL\_640\_50\_step30000.pt (trained using max num sentences = 640, max num word in sent = 50 for 30k steps)
