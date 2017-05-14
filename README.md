# GOTURN-Tensorflow

This is a tensorflow implementation of GOTURN.

Thanks to author **David Held** for his help of this implementation.

The original paper is: 

**[Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html)**,
<br>
[David Held](http://davheld.github.io/),
[Sebastian Thrun](http://robots.stanford.edu/),
[Silvio Savarese](http://cvgl.stanford.edu/silvio/),
<br>

The github repo for caffe implementation is given by the authors:
**[davheld/GOTURN](https://github.com/davheld/GOTURN)**

Brief illustration of how this network works:

<img src="imgs/pull7f-web_e2.png" width=85%>

You can refer to the paper or github repo above for more details.

## Environment
- python3
- tensorflow 1.0+, both cpu and gpu work fine

## How to use it
### Finetune for your own dataset
1. Create a folder, fill in all training images
2. Create a <train_txt_file>.txt file
    - It should contains target image, searching image and ground-truth bounding box
    - Bounding box is in the form of `<x1, y1, x2, y2>`, usually from 0 to 1, but exceeding this range is also fine.
    - Example of one line: 
    `train/target/000024.jpg,train/searching/000024.jpg,0.29269230769230764,0.22233115468409587,0.7991794871794871,0.7608061002178649`
3. Change related places in `train.py`
4. Train it and wait!
```python
python train.py
```
5. The log file is `train.log` by default

### Test 
1. Download pretrained model from: [GOTURN_MODEL](https://drive.google.com/open?id=0BwToyaMzz69QZ3Zlc0h4NzhBNDg)
2. Uncompress the `checkpoints` folder, and put it in the root directory of this repo 
3. Test on examples just by running `load_and_test.py` 
```python
python load_and_test.py
```
4. The log file is `test.log` by default

### TIPS
Be careful, the output of this network actually always from 0 to 10 thus I multiplied the ground-truth bounding boxes( always ranging from 0 to 1) by 10.

