This file is for the information about code and files for the work done in paper titled "Predictive Modeling of Critical Temperatures in Magnesium Compounds Using Transfer Learning."

Information about the files: 

The supplementary materials folder contains the following files:

	1. Transferlearning.py : Python file for training the transfer learning model.

	2. TrainSet : Contains critical temperature values and features for target dataset. 
	
	2. TestSet : Contains test dataset for critical temperature.

	3. Base_model.keras: Contains the base model trained with bandgap data.

	4. TL_modelTrained: Contains the finetuned transfer learning model for critical temperature prediction.  

	5. Ehull_Code.py: python file for training the Extra Trees Regressor for predicting energy above hull.

	6. Ehull_TrainSet: Contains Ehull values and features for stability dataset.

	7. Ehull_TestSet: Contains testset for stability dataset.

	8. Extra_Trees.sav: Contains the trained model for Ehull prediction. 
  
	9. OnlyMgPredictions.csv: Contains the predicted critical temperature for ~19k Mg compounds.

	10. All_Virtual_Materials.csv: Contains all predicted virtual compounds along with critical temperature and Ehull values.

	11. Stable_Virtual_Compounds: Contains only stable virtual compounds along with critical temperature and Ehull values..


	
	