!/bin/bash

cd AmbigCaps

# Tokenize 

export LC_ALL=en_US.UTF_8

# Set path to Moses clone
MOSES="../mmt_scripts/moses-3a0631a/tokenizer"
export PATH="${MOSES}:$PATH"

# Raw files path
RAW=raw
TOK=tok
SUFFIX="lc.norm.tok"

mkdir -p $TOK &> /dev/null

##############################
# Preprocess files in parallel
##############################
for TYPE in "train" "val" "test"; do
  for LLANG in "en" "tr"; do
    INP="${RAW}/${TYPE}.${LLANG}.gz"
    OUT="${TOK}/${TYPE}.${SUFFIX}.${LLANG}"
    if [ -f $INP ] && [ ! -f $OUT ]; then
      zcat $INP | lowercase.perl | normalize-punctuation.perl -l $LLANG | \
          tokenizer.perl -l $LLANG -threads 2 > $OUT &
    fi
  done
done
wait

# Apply BPE

# BPE related variables
BPE_MOPS=10000

TOK=tok
BPE=bpe${BPE_MOPS}

BPEPATH="../mmt_scripts/subword-nmt"
BPEAPPLY=${BPEPATH}/apply_bpe.py
BPELEARN=${BPEPATH}/learn_joint_bpe_and_vocab.py

SUFFIX="lc.norm.tok"

# Create folders
mkdir -p $BPE &> /dev/null

#####
# BPE
#####
for TLANG in "tr"; do
  LPAIR="en-${TLANG}"
  mkdir -p "${BPE}/${LPAIR}" &> /dev/null
  BPEFILE="${BPE}/${LPAIR}/codes"
  if [ -f $BPEFILE ]; then
    continue
  fi

  $BPELEARN -s $BPE_MOPS -o $BPEFILE \
    --input ${TOK}/train.${SUFFIX}.en \
            ${TOK}/train.${SUFFIX}.${TLANG} \
    --write-vocabulary \
            "${BPE}/${LPAIR}/vocab.en" "${BPE}/${LPAIR}/vocab.$TLANG" &
done
wait

# Apply for all pairs separately
for LPAIR in "en-tr"; do
  BPEFILE="${BPE}/${LPAIR}/codes"

  for TYPE in "train" "val" "test"; do
    # Iterate over languages
    for LLANG in `echo $LPAIR | tr '-' '\n'`; do
      INP="${TOK}/${TYPE}.${SUFFIX}.${LLANG}"
      OUT="${BPE}/${LPAIR}/${TYPE}.${SUFFIX}.bpe.${LLANG}"
      if [ -f $INP ] && [ ! -f $OUT ]; then
        echo "Applying BPE to $INP"
        $BPEAPPLY -c $BPEFILE --vocabulary \
          "${BPE}/${LPAIR}/vocab.${LLANG}" < $INP > $OUT &
      fi
    done
  done
done
wait

# Preprocess
TEXT=bpe10000/en-tr/
fairseq-preprocess \
    --source-lang tr --target-lang en \
    --trainpref $TEXT/train.lc.norm.tok.bpe \
    --validpref $TEXT/val.lc.norm.tok.bpe \
    --testpref $TEXT/test.lc.norm.tok.bpe \
    --destdir data-bin/ --joined-dictionary \
    --thresholdtgt 0 --thresholdsrc 0 --workers 20