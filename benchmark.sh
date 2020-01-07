python tiresias/benchmark/regression.py
python tiresias/benchmark/classification.py
python tiresias/benchmark/finetuning.py --epochs 16 --epsilon 1.0   --feature_extraction 1
python tiresias/benchmark/finetuning.py --epochs 16 --epsilon 10.0  --feature_extraction 1
python tiresias/benchmark/finetuning.py --epochs 16 --epsilon 100.0 --feature_extraction 1
python tiresias/benchmark/finetuning.py --epochs 16 --epsilon 1.0   --feature_extraction 0
python tiresias/benchmark/finetuning.py --epochs 16 --epsilon 10.0  --feature_extraction 0
python tiresias/benchmark/finetuning.py --epochs 16 --epsilon 100.0 --feature_extraction 0
