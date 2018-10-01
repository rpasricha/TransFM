# TransFM

This repository contains an implementation of TransFM, as described in the paper:

Rajiv Pasricha, Julian McAuley,
"Translation-based Factorization Machines for Sequential Recommendation",
RecSys 2018.

This repository also includes implementations of vanilla FMs, as well as the proposed
PRME-FM and HRM-FM models.

Please cite the paper above if you use or extend our models.


## File formats
- Input dataset
    - One example per line
    - `<user_id> <item_id> <rating> <timestamp>`
    - Values separated by a space
    - No header row
    - Example row: `User_12 Item_65 5.0 1376697600`

- Item categories
    - CSV file, one item per line
    - Expected header: `item_id,item_cat_seq`
    - `item_cat_seq`: comma separated list of item category IDs, enclosed as a string.
    - Example row: `2643,"[165, 193, 442]"`

- User features
    - CSV file with numeric features, one user per line
    - Header row expected, first column should be named `idx`
    
- Item features
    - CSV file with numeric features, one item per line
    - Header row expected, first column should be named `idx`

- Geographical features
    - CSV file with numeric features, one item per line
    - Header row expected, first column should be named `place_id`

## Example command
```
python main.py \
      --filename ratings_Automotive.txt.gz
      --model TransFM
      --features categories
      --features_file item_cat_seq_Automotive.csv.gz
      --max_iters 1000000
      --num_dims 10
      --linear_reg 10.0
      --emb_reg 1.0
      --trans_reg 0.1
      --init_mean 0.1
      --starting_lr 0.02
      --lr_decay_factor 1.0
      --lr_decay_freq 1000
      --eval_freq 50
      --quit_delta 1000
```
