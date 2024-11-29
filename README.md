**Capsule-Vision-2024-Challenge**

This repository contains the implementation of the Capsule Vision 2024 challenge. The project is structured into three main folders: Train, Test, and Validation, each containing Python scripts to handle the corresponding tasks.

**Folder Structure**

Train/ :
Contains the script train.py for training and validating the model.

Test/ :
Contains the script test.py for testing and evaluating the model's performance on unseen data.

Validation/ :
Contains the script valid.py for getting the evaluation matrics on validation data.

**Setup Instructions**

To get started, follow these steps:

1. Clone the repository:
git clone https://github.com/ajay-pratap-singh-518/Capsule-Vision-2024-challenge.git

2. Navigates into the folder:
cd Capsule-Vision-2024-Challenge

3. Install the required dependencies: Ensure you have the necessary Python packages installed by running:-
pip install -r requirements.txt

4. Run the training script: To train the model, execute:
python train/train.py

5. Run the validation script: For validation during training, use:
python valid/valid.py

6. Run the testing script: Once the model is trained, you can evaluate it using:
python valid/valid.py

**Achievement**
We are happy to announce that our team ranked 13th in the Capsule Vision 2024 Challenge, with 150 teams participating globally. Out of these, only 35 teams submitted their entries, and after review, 27 teams were selected for evaluation. We thank the organizers for this opportunity to showcase our model for multi-class classification in video capsule endoscopy.

**Citations**

Challenge ArXiv

@article{handa2024capsule, title={Capsule Vision 2024 Challenge: Multi-Class Abnormality Classification for Video Capsule Endoscopy}, author={Handa, Palak and Mahbod, Amirreza and Schwarzhans, Florian and Woitek, Ramona and Goel, Nidhi and Chhabra, Deepti and Jha, Shreshtha and Dhir, Manas and Gunjan, Deepak and Kakarla, Jagadeesh and others}, journal={arXiv preprint arXiv:2408.04940}, year={2024}}

Training and Validation Datasets

@article{Handa2024, author = "Palak Handa and Amirreza Mahbod and Florian Schwarzhans and Ramona Woitek and Nidhi Goel and Deepti Chhabra and Shreshtha Jha and Manas Dhir and Deepak Gunjan and Jagadeesh Kakarla and Balasubramanian Raman", title = "{Training and Validation Dataset of Capsule Vision 2024 Challenge}", year = "2024", month = "7", url = "https://figshare.com/articles/dataset/Training_and_Validation_Dataset_of_Capsule_Vision_2024_Challenge/26403469", doi = "10.6084/m9.figshare.26403469.v1", journal={Fishare}}

Testing Datasets

@article{Handa2024, author = "Palak Handa and Amirreza Mahbod and Florian Schwarzhans and Ramona Woitek and Nidhi Goel and Deepti Chhabra and Shreshtha Jha and Manas Dhir and Pallavi Sharma and Dr. Deepak Gunjan and Jagadeesh Kakarla and Balasubramanian Ramanathan", title = "{Testing Dataset of Capsule Vision 2024 Challenge}", year = "2024", month = "10", url = "https://figshare.com/articles/dataset/Testing_Dataset_of_Capsule_Vision_2024_Challenge/27200664", doi = "10.6084/m9.figshare.27200664.v1" }

