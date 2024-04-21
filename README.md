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

#### Binding task.
```
# Generate binding trials with 8 objects.
python gen_binding_trials.py \
--n_objects 2 3 4 5 6 7 8 \      
--n_trials=100 \
--size=45 \
--color_names red green blue gold purple saddlebrown gray black \
--shape_names triangle cloud cross down-arrow umbrella pentagon heart star \
--shape_inds 9 21 24 28 34 59 96 98 \
--output_dir=data/binding_balanced

# Run model on binding trials.
python run_requests.py \
--task_dir='data/binding_balanced' \
--task_prompt_path='prompts/run/binding.txt' \
--parse_prompt_path='prompts/parse/binding.txt' \
--results_file='output/binding_balanced.csv' \
--max_tokens=500 \
--n_trials=20 \
--api='azure'
```


#### Popout task.
```
# Generate popout trials with only green and (one) red circles.
python gen_popout_trials.py \
--n_objects 5 10 15 20 25 30 35 40 45 50 \
--n_trials=100 \
--size=24 \
--colors green red \
--shape_inds 37 \
--output_dir=data/popout_simple

# Generate popout trials with circles of any colors.
python gen_popout_trials.py \
--n_objects 5 10 15 20 25 30 35 40 45 50 \
--n_trials=100 \
--size=22 \
--shape_inds 37 \
--output_dir=data/popout_complex

# Run models on popout trials.
#### Fill in here ####
```

#### Serial search task.
```
# Generate serial search trials with only green and red letters (T's and L's).
python gen_search_trials.py \
--n_objects 5 10 15 20 25 30 35 40 45 50 \
--n_trials=100 \
--size=24 \
--use_letters=True \
--colors green red \
--output_dir=data/search_letters

# Generate search trials with circles of any colors.
python gen_search_trials.py \
--n_objects 4 6 8 10 16 32 \
--n_trials=100 \
--size=24 \
--shape_inds 1 37 \
--output_dir=data/search_shapes

# Run models on serial search trials.
python run_requests.py \
--task_dir='data/search_letters' \
--task_prompt_path='prompts/run/search.txt' \
--parse_prompt_path='prompts/parse/search.txt' \
--results_file='output/letters_search.csv' \
--max_tokens=500 \
--api='azure'
```