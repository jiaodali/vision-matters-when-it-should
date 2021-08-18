# vision-matters-when-it-should

This repository accompanies the paper: [Vision Matters When It Should: Sanity Checking Multimodal Machine Translation Models]().

## Dependencies

- python: 3.7.4
- perl: 5.18.4
- pytorch: 1.7.1+cu101
- ```bash
  pip install -e .
  pip install -r requirements.txt
  ```

## Data Preparation
We use AmbigCaps dataset as an example. 
1. Run `mmt_scripts/download_dataset.sh` to download and decompress the dataset.

2. Run `mmt_scripts/prepare_text.sh` to prepare the textual data.

3. Run `mmt_scripts/prepare_image.sh` to extract visual features for the images. 

## Training Text-Only Transformer
The last ten checkpoints are averaged.
```
export CHECKPOINT_DIR=checkpoints/
export DATA_DIR=AmbigCaps/
fairseq-train $DATA_DIR/data-bin/ --task translation \
        --arch transformer_tiny --share-all-embeddings --dropout 0.3 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
        --lr 0.005 \
        --patience 10 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
        --max-tokens 4096\
        --update-freq 1 --no-progress-bar --log-format json --log-interval 100 \
        --save-interval-updates 1000 --keep-interval-updates 1000 --source-lang tr --target-lang en \
        --save-dir $CHECKPOINT_DIR/ \
        --find-unused-parameters \

python scripts/average_checkpoints.py \
        --inputs $CHECKPOINT_DIR/ \
        --num-epoch-checkpoints 10 \
        --output $CHECKPOINT_DIR/averaged_model.pt
```

## Training Multimodal Transformer
```
export CHECKPOINT_DIR=checkpoints/mmt/
export DATA_DIR=AmbigCaps/
fairseq-train $DATA_DIR/data-bin/ --task multimodal_translation \
        --arch multimodal_transformer_tiny --share-all-embeddings --dropout 0.3 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 2000 \
        --lr 0.005 \
        --patience 10 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0 \
        --max-tokens 4096\
        --update-freq 1 --no-progress-bar --log-format json --log-interval 100 \
        --save-interval-updates 1000 --keep-interval-updates 1000 --source-lang tr --target-lang en \
        --save-dir $CHECKPOINT_DIR/ \
        --find-unused-parameters \
        --word_dropout 0.1 \
        --fusion_method gated \
        --train_image_embedding_file $DATA_DIR/train-resnet50-avgpool.npy \
        --val_image_embedding_file $DATA_DIR/val-resnet50-avgpool.npy \
        --test_image_embedding_file $DATA_DIR/test-resnet50-avgpool.npy \


python scripts/average_checkpoints.py \
        --inputs $CHECKPOINT_DIR/ \
        --num-epoch-checkpoints 10 \
        --output $CHECKPOINT_DIR/averaged_model.pt
```
- With `word_dropout`, tokens in the source sentence are dropped randomly, subject to the given dropout probability. You can leave out this argument to turn it off. 
- There are two approaches for fusing textual and visual features: `gated` and `concat`, which can be specified with the argument `fusion_method`.
- `{train,val,test}_image_embedding_file` point to the locations of pre-extracted visual features. Note that `gated` fusion requires `avgpool` features while `concat` requires `res4frelu` features.

## Generatoin

```
export CHECKPOINT_DIR=checkpoints/mmt/
export DATA_DIR=AmbigCaps/
export RESULT_DIR=results/
mkdir -p $RESULT_DIR
fairseq-generate $DATA_DIR/data-bin/  \
    --task multimodal_translation \
    --source-lang tr --target-lang en \
    --path $CHECKPOINT_DIR/averaged_model.pt \
    --shuffle_image \
    --train_image_embedding_file $DATA_DIR/train-resnet50-avgpool.npy \
    --val_image_embedding_file $DATA_DIR/val-resnet50-avgpool.npy \
    --test_image_embedding_file $DATA_DIR/test-resnet50-avgpool.npy \
    --beam 5 --batch-size 128 --remove-bpe | tee $RESULT_DIR/gen.out
```
With `shuffle_image`, images in a batch will be shuffled. You can also specify the random seed with `seed` argument.

## Acknowledgments 
- Our codes are inspired by [UVR-NMT](https://github.com/cooelf/UVR-NMT), which is based on [Fairseq](https://github.com/pytorch/fairseq).
- The data preparation scripts are borrowed from [Multi30k](https://github.com/multi30k/dataset) to minize processing differences across datasets.