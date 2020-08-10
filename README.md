# Context-Aware Model on Fairseq

The implementation of "Does Multi-Encoder Help? A Case Study on Context-Aware Neural Machine Translation"

> This code is based on [Fairseq v0.6.2](https://github.com/pytorch/fairseq/tree/v0.6.2)

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6

1. `pip3 install -r requirements.txt`
2. `python3 setup.py develop`
3. `python3 setup.py install`

## Prepare Training Data

> `bash runs/prepare-en2ru.sh`

## Train

### Train transformer baseline

> `bash runs/train-en2ru.sh baseline`

### Train context-aware model

> `bash runs/train-en2ru.sh inside-context`
>
> `bash runs/train-en2ru.sh outside-context`

### Train model with gaussian noise

> `bash runs/train-en2ru.sh gaussian`

## Infer

> `bash runs/translate-en2ru.sh baseline`
>
> `bash runs/translate-en2ru.sh inside-context`
>
> `bash runs/translate-en2ru.sh outside-context`
>
> `bash runs/translate-en2ru.sh gaussian`

## Infer without context

> `bash runs/translate-en2ru.sh inside-context ignore`
>
> `bash runs/translate-en2ru.sh outside-context ignore`

## Citation

```bibtex
@inproceedings{li-etal-2020-multi,
    title = "Does Multi-Encoder Help? A Case Study on Context-Aware Neural Machine Translation",
    author = "Li, Bei  and
      Liu, Hui  and
      Wang, Ziyang  and
      Jiang, Yufan  and
      Xiao, Tong  and
      Zhu, Jingbo  and
      Liu, Tongran  and
      li, changliang",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.322",
    pages = "3512--3518",
}
```
