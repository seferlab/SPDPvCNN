# **_SPDPvCNN_**

**_`Stock Price and Direction Prediction via Deep Attention-Based Convolutional Neural Networks`_**

**_This is the repository for the "Stock Price and Direction Prediction via Deep Attention-Based Convolutional Neural Networks" CS 401 and CS 402 Senior Project at Ozyegin University. This is where you'll find all of the essential materials, such as code, data, and articles._**

## **_Dependencies Used_**

**_In order to run `convmixer.py`, `vision_transformer.py`, and `mlp_mixer.py` one must have the exact versions of dependencies below._**<br/>
**_Note that the code is incompatible with versions of tensorflow 2.5.0, 2.7.0, and 2.8.0._**

- **_[Keras:](https://keras.io/) 2.6.0_**
- **_[Tensorflow:](https://www.tensorflow.org/) 2.6.0_**
- **_[Tensorflow Addons:](https://www.tensorflow.org/addons) 0.16.1_**

## **_Steps to Re-Produce Our Results_**

**_1. Clone the Repository to Your Local Machine_**

```bash
git clone https://github.com/kuantuna/SPDPvCNN.git
```

**_2.0 Make sure that your folder structure in the ETF folder looks exactly like the figure below. (By creating the necessary folders)<br/>_**
![Figure](https://github.com/kuantuna/SPDPvCNN/blob/main/images/folder_structure.png?raw=true)<br/>
**_2.1 Run the `data_creation.py` after setting the threshold value (default = 0.01) to create images and labels for the specified range of dates.<br/>_**
**_3. Later, in the `architectures/helpers/constants.py` choose the architecture (selected_model) you want to use (by commenting others) and run the `training.py` to train the model using the images created on the previous phase.<br/>_**
**_4. Finally, run the `financial_evaluation.py` and `computational_evaluation.py` to evaluate our model on the test data both financially and computationally.<br/>_**

## **_Institution_**

- **_[Ozyegin University](https://www.ozyegin.edu.tr/)_**

## **_Project Members_**

- **_[Tuna Tuncer](https://github.com/kuantuna)_**<br/>
- **_[Uygar Kaya](https://github.com/UygarKAYA)_**<br/>
- **_[Onur Alaçam](https://github.com/Onralcm)_**<br/>
- **_[Tuğcan Hoşer](https://github.com/Tugcannn)_**

## **_Project Supervisor_**

- **_[Assistant Prof. Emre Sefer](http://www.emresefer.com/)_**
