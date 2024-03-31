
* This app was developed on a Linux system and has not been tested on Windows.
# **Music Genre Classification**

This project is aimed at classifying music genres using machine learning techniques. It involves several Python scripts that perform various tasks such as data preprocessing, feature extraction, model training, and prediction.

The best model for music genre classification was found using grid search beforehand. Additionally, CSV files containing precomputed features were used for training the models. The dataset used in this project includes the Gtazen dataset along with a custom dataset i've created.

The results for the models I experimented with are as follows: SVM (93%), KNN (92%), Adaboost (85%), and Perceptron (87%).Trained machine learning models with SVM (93%), 

# **Model-View-Controller (MVC) Architecture**

This application follows the Model-View-Controller (MVC) architectural pattern, which helps in organizing code and separating concerns. Here's how the MVC components are utilized in this project:

- **Model**: Handles the core functionality of the application, including audio loading, feature extraction, genre prediction, and audio recording. It encapsulates the application's logic and maintains its state.

- **View**: Provides the graphical user interface (GUI) for the application. It presents visual elements such as buttons, progress bars, and text output to the user. The view updates in response to changes in the model and user interactions.

- **Controller**: Acts as an intermediary between the model and the view. It translates user inputs into actions that the model can understand and updates the view accordingly. The controller orchestrates the flow of data and events within the application.

The MVC architecture promotes modularity, scalability, and maintainability by separating the presentation logic (view), application logic (model), and user input handling (controller). This design pattern facilitates code reuse, enhances testability, and improves overall code organization.


MVC Application User-End:
  
![Screenshot 2024-03-19 at 21 55 45](https://github.com/barrShahar/MusicGnereApp/assets/59974036/735b0cd8-992c-4aa9-a399-24cd7151cd7d)

![Screenshot 2024-03-19 at 21 56 46](https://github.com/barrShahar/MusicGnereApp/assets/59974036/22edbddd-8054-4f0d-8568-88a258f434df)
