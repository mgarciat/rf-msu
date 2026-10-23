# rf-msu
This project contains the code to run the MSU-based Random Forest (RFMSU). Please, cite the work if you use it, thank you:

García-Torres, M., Saucedo, F., Divina, F., & Gómez-Guerrero, S. (2025). RFMSU: A multivariate symmetrical uncertainty-based random forest. Pattern Recognition, 111939.

# code
This folder contains the RFMSU and RandomTreeMSU codes. Both classes have to be added into the weka/classifiers/tree folder and add the libraries that you will find at lib folder.

A python version of RFMSU is also available under python folder and python_pip with an example of how to run it. To run a similar version of the paper, run i with the following configuration:
rf = RandomForestMSU(
    numIterations=100,
    kValue=0,
    minNum=1.0,
    maxDepth=0,
    bagSizePercent=100,
    bootstrap_mode="weighted",
    rediscretize_msu_per_node=True,
    breakTiesRandomly=False,
    seed=1,
    n_jobs=1,
    debug=False
) 
# lib
This folder contains the libraries necessary to run RFMSU class.
# project
Contains a Netbeans project to see an example of how to run RFMSU. 
