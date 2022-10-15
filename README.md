# blog.deep-metric-learning
An exploration into contrastive and triplet deep metric learning methods

This project is an attempt for me to learn more about the field of deep metric learning, and share some of this knowledge through a blog post. The blog post is hosted via github pages and can be found [here](https://frans-db.github.io/blog.deep-metric-learning/)

Install Required packages
```
pip install torch torchvision matplotlib imageio
```

Running:
```
python metric_learnign/main.py
```

Optional Arguments (each of these has a default and is not needed)
```
  -h, --help            show this help message and exit
  --results_root RESULTS_ROOT
                        Name of the directory to store experiments in
  --experiment_name EXPERIMENT_NAME
                        Name of the current experiment. Used to store results
  --mode MODE           Mode to use. contrastive (default) or triplet
  --labels LABELS [LABELS ...]
                        Labels to use for the experiment
  --dimensionality DIMENSIONALITY
                        Manifold dimensionality to map the data to
  --epochs EPOCHS       Number of training epochs
  --test_every TEST_EVERY
                        Number of training epochs
  --repeat_frames REPEAT_FRAMES
                        Repeat a frame a number of times to slow down the GIF
  --repeat_last_frame REPEAT_LAST_FRAME
                        Repeat the last frame a number of times to pause the GIF here
  --batch_size BATCH_SIZE
                        Dataloader batch size
  --num_workers NUM_WORKERS
                        Dataloader number of workers
```
