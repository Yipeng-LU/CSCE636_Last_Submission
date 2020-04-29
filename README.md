# CSCE636_Last_Submission
To run the prediction function:
1. Open colab in GPU mode
2. Copy everything in Install_Openpose and run in colab
3. Copy everything in Prediction_Function and run in colab
4. Upload the video onto colab root directory('/content/')
5. Upload the model onto colab root directory: new_model20.h5
5. Call the function in colab: predAndPlot(root), where the variable root is root of the video.
6. There are also some screenshots of actions that my model learn, which are drinking, walking and playing with pad(3 non-stretching actions) vs. sky reaching, stretching leg and waist twisting(3 stretching actions). These 6 actions under both standing and sitting condition from various of angles are learnt by my model.
