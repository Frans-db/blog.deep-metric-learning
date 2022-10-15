# blog.deep-metric-learning
An exploration into contrastive and triplet deep metric learning methods

This project is an attempt for me to learn more about the field of deep metric learning, and share some of this knowledge through a blog post. The blog post is hosted via github pages and can be found [here](https://frans-db.github.io/blog.deep-metric-learning/).

During this project I set myself a deadline of 24 hours to finish everything. Usually I quite often leave projects unfinished, and the goal behind this deadline was to finally finish something I start on. I think this idea of such 24 hour sprints has been discussed in multiple places, but I got the idea from [this youtube video](https://www.youtube.com/watch?v=AIr9GeVzHRw) by [Folding Ideas](https://www.youtube.com/c/FoldingIdeas).

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
