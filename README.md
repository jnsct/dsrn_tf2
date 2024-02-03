# Image Super-resolution via Dual-state Recurrent Neural Networks (CVPR 2018), Tensorflow 2
### [[Paper Link]](https://arxiv.org/pdf/1805.02704.pdf)

### Citation

	@inproceedings{han2018image,  
		title={Image super-resolution via dual-state recurrent networks},
		author={Han, Wei and Chang, Shiyu and Liu, Ding and Yu, Mo and Witbrock, Michael and Huang, Thomas S},
		booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
		year={2018}
	}
  
### Dependencies
- Common python dependencies can be installed via `pip install -r requirements.txt`
- Lingvo is no longer required for weights

### Data
There is a very helpful repo [collected](https://github.com/jbhuang0604/SelfExSR#datasets) download links for all the training and test sets needed here. 
### Training 
The training data is specified by a file list of HR images. No futher pre-processing is needed as we perform downsampling and augmentation on-the-fly.

Use `train.py` and the model specification file `model_recurrent_s2_u128_avg_t7.py` to start a training job. 

### Inference
Models are not provided, and must be trained by the user using the train.py file.

### Evaluation
Use `evaluate.py` to compute average PSNR on a test set after saving all the model predicted images.

### Acknowledgement
This code is partly based on a previous work from the group [[here]](https://github.com/ychfan/sr_ntire2017), as well as the original DSRN project

