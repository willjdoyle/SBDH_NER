William Doyle
COMP.5520-201 Foundations in Digital Health
Project #3 - SBDH NER
Due 4/15/23

===========
GitHub Link
===========
https://github.com/willjdoyle/SBDH_NER.git

===================
Program Description
===================
This set of Python programs will load the specified notes in the given train.csv from UML's MIMIC III database, and create, train, and evaluate a spaCy NER pipeline.

===================
Runtime Environment
===================
This program was run in Visual Studio using Python 3.8.10 in Windows 10.0.19045. The training and evluation portions of the programs also took advantage of my graphics card using NVIDIA CUDA Development Toolkit.
The version for the NVIDIA (R) Curda compiler driver is: release 11.8, V11.8.89.

For the Python scripts, the following libraries and versions were used:

os (-)
random (-)
warnings (-)
math (-)
spacy (3.5.1)
psycopg2 (2.9.5)
pandas (1.2.4)
tqdm (4.65.0)

=============
Project Files
=============
train.csv - Given data for this assignment. Includes the row_id, label, and location of labeled entity in that row_id's note text.
Load_Data.py - Establishes connection with UML's MIMIC III DB, preprocesses data and loads into a training and testing spaCy DocBin. Uses an 80/20 train/test split. Note that this program can take up to one minute to run. Creates a train.spacy and test.spacy file in the same directory.
Train_Model.py - Loads the train.spacy file, trains a model, and then saves the model and nlp vocabulary as 'ner_based_on_scibert' and 'new_scibert' in the same directory. See the "Hyperparameters" section of this README for more information on tuning the model.
Test_Model.py - Loads the specified model and test.spacy to test the model. Alternatively, can use the spaCy CLI.
Data_Visualizer.py - Simple extra program to show the entities of a given Doc. Specify either train.spacy or test.spacy on line 25, and then specify DOC_SELECTION on line 38 for the specific Doc you'd like to see, and then to go localhost:5000 to see the labeled sentence.

train.spacy, test.spacy - Files created by Load_Data.py and used by Train_Model.py. They are spaCy DocBin files.

===============
Hyperparameters
===============
n_iter - Number of iterations (times the training loop will run). Generally, a higher number of iterations means it will train for longer but result in lower loss.
batch_size - How big the training batches are. A higher number will allow the model to train faster, but will cause an out-of-memory (OOM) error if it is too large.
compounding(x,y,z) - Used for determining the minibatch size.
drop - Determines the drop rate. Higher values might allow loss to decrease faster and avoid local minima, but might also accidentally overshoot any minima.

For more details on the hyperparameter optimization, see the "Hyperparameter Optimization Results" section at the end of this README.

=====================
Pipeline/Result Files
=====================
ModelResults.xlsx - Spreadsheet containing accuracies of several models that were created and tested.
new_scibert_BEST, ner_based_on_scibert_BEST - The best performing model and its vocab (based on F-score) that I had trained.

***NOTE***
ModelResults.xlsx lists other tests. They have not been included in this submission to reduce the download size. Please let me know if you would like them, and I will happily add them to the repository.

====================
Runtime Instructions
====================
1) First, make sure you are connected to the UML VPN. Download the files and place them in a new folder.
2) Run Load_Data.py to load and preprocess the data.
3) Run Train_Model.py (after tuning the hyperparameters in the training section of the code) and wait for it to complete. Using an an RTX 2070 GPU, I've found that 100 iterations with the current parameters will take around 60 - 90 minutes.
4) Run Test_Model.py (after specifying the model in the spacy.load() function) to test the model. If you would prefer, you can use the spaCy terminal command (in the same directory): spacy evaluate .\ner_based_on_scibert_BEST\ test.spacy --gpu-id 0

===================================
Hyperparameter Optimization Results
===================================
Below are a summary list of the various tests I ran, and I selected the most notable from the list.

Test 1: n_iter=100, batch_size=1024, compounding(4.0,32.0,1.001) - Loss decreased from 8,000 to 5,500.

Test 2: n_iter=100, batch_size=3000, compounding(4.0,32.0,1.001) - Loss decreased from 8,000 to 5,100.

Test 3: n_iter=100, batch_size=3000, compounding(4.0,32.0,1.01)  - Loss decreased from 8,000 to 5,000.

Test 4: n_iter=750, batch_size=3000, compounding(4.0,32.0,1.01)  - Loss decreased from 8,000 to 3,350.

Test 5: n_iter=100, batch_size=3000, compounding(4.0,16.0,1.01)  - Loss decreased from 8,000 to 4,800.

Test 6: n_iter=100, batch_size=3000, compounding(8.0,16.0,1.01)  - Loss decreased from 8,000 to 4,900.

Test 7: n_iter=100, batch_size=3000, compounding(1.0,16.0,1.01)  - Loss decreased from 8,000 to 5,200.

Test 8: n_iter=100, batch_size=3000, compounding(4.0,32.0,1.01)  - Loss decreased from 8,000 to 4,900.

Test 9: n_iter=100, batch_size=3000, compounding(4.0,64.0,1.01)  - Loss decreased from 8,000 to 4,900.

Test 10: n_iter=100, batch_size=3000, compounding(4.0,32.0,1.1)  - Loss decreased from 8,000 to 4,900.

Test 11: n_iter=750, batch_size=3000, compounding(4.0,16.0,1.01) - Loss decreased from 8,000 to 3,400.

From the findings above, it seemed that the best parameters were those used in Test 8. Expanding on this test by changing the drop rate gave:

Test 8.1: drop=0.8   - Loss decreased from 10,000 to 7,200.

Test 8.2: drop=0.2   - Loss decreased from 7,000 to 2,280.

Test 8.3: drop=0.01  - Loss decreased from 7,000 to 800.

Test 8.4: drop=0.001 - Loss decreased from 6,000 to 800.