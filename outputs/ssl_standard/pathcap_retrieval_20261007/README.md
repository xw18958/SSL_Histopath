# PathCap retrieval

Both pretrained SSL methods were evaluated at all five saved checkpoints using one fixed 3,267 TRAIN / 700 VAL / 700 TEST sample. Only the linear image projection heads were trained. Paper IDs and identical captions were kept within splits. The one empty-caption row and 577 rows without a known paper identifier were excluded before sampling. All ten heads were frozen before TEST began. No SSL pretraining or earlier dataset experiments were rerun.

The four NVIDIA GPUs across the three servers were used by independent workers. The additional Intel graphics device on gpu1-358-0 was not CUDA compatible.
