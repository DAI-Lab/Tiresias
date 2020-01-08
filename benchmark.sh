python tiresias/benchmark/regression.py
python tiresias/benchmark/classification.py
python tiresias/benchmark/finetuning.py --epochs 64 --epsilon 8.0   --feature_extraction 1
python tiresias/benchmark/finetuning.py --epochs 64 --epsilon 16.0  --feature_extraction 1
python tiresias/benchmark/finetuning.py --epochs 64 --epsilon 32.0 --feature_extraction 1
python tiresias/benchmark/finetuning.py --epochs 64 --epsilon 8.0   --feature_extraction 0
python tiresias/benchmark/finetuning.py --epochs 64 --epsilon 16.0  --feature_extraction 0
python tiresias/benchmark/finetuning.py --epochs 64 --epsilon 32.0 --feature_extraction 0
