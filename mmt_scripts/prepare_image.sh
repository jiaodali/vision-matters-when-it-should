!/bin/bash
cd AmbigCaps
for split in "val" "test" "train"; do
	python ../mmt_scripts/feature-extractor -f ${split}/ -b 8
done