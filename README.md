# DLCV Final Project ( Face Anti-spoofing )

# How to run your code?
> 1. bash FRANK.sh
> 2. Results will be saved in oulu.csv siw.csv bonus.csv
    
# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2020/faceantispoofing-<team_name>.git
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1T8Wh9rM5zCiuMVCulDCZwX9JZZ9Mqgd0Yr3uqgPpe1I/edit?usp=sharing) to view the slides of Final Project - Face Anti-Spoofing. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided 2 shell scripts for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset_SiW.sh
    bash ./get_dataset_oulu.sh
The shell scripts will automatically download the dataset and store the data. Note that these commands by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link_oulu](https://drive.google.com/file/d/1251SwV6bnMrDF0EZ8kdmH2FZQTLSLbgU/view) & [this_link_SiW](https://drive.google.com/file/d/1eUd3Y0_9y_xZ6CDDn3p9Y2KWKH997do_/view) and unzip the compressed files manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `oulu` and a folder called `SiW` under the root directory of your local repository so that they will be included in the default `.gitignore` file.

### Evaluation
We will use AUC to evaluate your model. Please refer to the introduction ppt for more details.

# Submission Rules
### Deadline
2021/1/22 11:59 GMT+8

### Late Submission Policy
#### Late Submission is NOT allowed for final project!

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      


> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to `requirments.txt` for more details.

You can run the following command to install all the packages listed in `requirements.txt`:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.


# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in FB group
