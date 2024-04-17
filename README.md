# vlm-binding

#### Counting task
```
# Generate counting trials with randomly colored circles. Saves results and metadata to output_dir.
python gen_counting_trials.py --object_inds=37 \
                              --n_trials=100 \
                              --sigma=1 \
                              --n_shapes 1 2 3 4 5 6 7 8 9 10 \
                              --size=45 \
                              --output_dir=data/counting_high_variance

# Generate counting trials with colored circles with low variance. Saves results and metadata to output_dir.
python gen_counting_trials.py --object_inds=37 \
                              --n_trials=100 \
                              --sigma=0.01 \
                              --n_shapes 1 2 3 4 5 6 7 8 9 10 \
                              --size=45 \
                              --output_dir=data/counting_low_variance

# Generate counting trials with only black circles. Saves results and metadata to output_dir.
python gen_counting_trials.py --object_inds=37 \
                              --n_trials=100 \
                              --n_shapes 1 2 3 4 5 6 7 8 9 10 \
                              --size=45 \
                              --all_black=True \
                              --output_dir=data/counting_black

# Run model on counting trials.
#### Fill in here ####
```


```
--n_objects 2 3 4 5 6 7 8 9 10 

python gen_binding_trials.py \
--n_objects 8 \
--n_trials=100 \
--size=45 \
--color_names red green blue gold purple saddlebrown gray black \
--shape_names triangle cloud cross down arrow umbrella pentagon heart star \
--shape_inds 9 21 24 28 34 59 96 98 \
--output_dir=data/binding
```