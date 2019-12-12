#!/bin/bash
#Example to Run the multimodal model code on the Multi30K DE Dataset 
python nmt_multimodal_beam_DE.py --data_path data/Multi30K_DE \
								 --trained_model_path /path/to/save/model \
								 --sr en \
								 --tg de

#Example to Run the monomodal model code on the Multi30K DE Dataset 
python nmt_monomodal_beam_DE.py --data_path data/Multi30K_DE \
								 --trained_model_path /path/to/save/model \
								 --sr en \
								 --tg de