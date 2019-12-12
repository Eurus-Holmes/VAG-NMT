import torch
import numpy as np

def t2i(images, captions):
	"""
	Text -> Image
	Images: (N,K) matrix of images
	Captions: (N,K) matrix of cpations
	"""
	npts = images.shape[0] #Define the number of images

	#Initialize the ranks
	ranks = np.zeros(npts)
	
	for index in range(npts):

		#Get query captions
		queries = captions[index].unsqueeze(0)

		#Compute Scores
		d = torch.mm(queries, images.t())
		d_sorted, inds = torch.sort(d, descending=True)
		inds = inds.squeeze(0).cpu().numpy()
		ranks[index] = np.where(inds == index)[0][0]
	#compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	return (r1, r5, r10, medr)

def i2t(images, captions):
	"""
	Text -> Image
	Images: (N,K) matrix of images
	Captions: (N,K) matrix of cpations
	"""
	npts = images.shape[0] #Define the number of images

	#Initialize the ranks
	ranks = np.zeros(npts)
	
	for index in range(npts):

		#Get query captions
		queries = images[index].unsqueeze(0)

		#Compute Scores
		d = torch.mm(queries, captions.t())
		d_sorted, inds = torch.sort(d, descending=True)
		inds = inds.squeeze(0).cpu().numpy()
		ranks[index] = np.where(inds == index)[0][0]
	#compute metrics
	r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
	r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
	r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
	medr = np.floor(np.median(ranks)) + 1
	return (r1, r5, r10, medr)