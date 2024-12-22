# One-step multi-view clustering guided by weakened view-specific distribution
## Environment installation
First install the training/testing environment via the following commands.

``
pip install -r requirements.txt -i https://pypi.douban.com/simple
``
## Train
First, you need to set the data_path parameter value in args. Just set it to the path where the training data file is located, for example: "./data".

Run the following command to train the model.

``
python train.py
``
## Test
If you just want to view the trained model, we provide the trained model, you only need to run the test.py file.

``
python test.py
``

## Data
Due to Github upload restrictions, the training data and trained models we used are uploaded to Baidu Cloud disk for download.

Download url:
https://pan.baidu.com/s/13KIBLq8nQkWEEplVw656DA?pwd=dc59 password: dc59
## Citation
If you found our work or code useful, please cite this work as follows, thank you.
```c
@article{cai2024one,
  title={One-step multi-view clustering guided by weakened view-specific distribution},
  author={Cai, Yueyi and Wang, Shunfang and Wang, Junjie and Fei, Yu},
  journal={Expert Systems with Applications},
  pages={124021},
  year={2024},
  publisher={Elsevier}
}
```
