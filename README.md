# BibleSentiment
Sentiment Analysis of the four Gospels

Use this link to launch a dockerized version of this project. This will allow you to run the code in the browser and see the results of the analysis.
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wthegamer/BibleSentiment/main)

Here are the steps you’ll need to take before running the notebooks locally(on your own machine) in this project:

1. Download Anaconda
I highly recommend that you download Python at at least version 3.7. At the time of writing this I am using 3.11.5

2. Download the Jupyter Notebooks
Clone or download this Github repository, so you have access to all the Jupyter Notebooks (.ipynb extension) in the tutorial. Note the green button on the right side of the screen that says Clone or download. If you know how to use Github, go ahead and clone the repo. If you aren't comfortable using git or GitHub try clicking the launch link up above.

3. Launch Anaconda and Open a Jupyter Notebook
Windows: Open the Anaconda Navigator program. You should see the Jupyter Notebook logo. Below the logo, click Launch. A browser window should open up. In the browser window, navigate to the location of the saved Jupyter Notebook files and open 0-Hello-World.ipynb. Follow the instructions in the notebook.

Mac/Linux: Open a terminal. Type jupyter notebook. A browser should open up. In the browser window, navigate to the location of the saved Jupyter Notebook files and open 0-Hello-World.ipynb. Follow the instructions in the notebook.

4. Install a Few Additional Packages
There are a few additional packages we'll be using during the tutorial that are not included when you download Anaconda - wordcloud, textblob and gensim.

Windows: Open the Anaconda Prompt program. You should see a black window pop up. Type conda install -c conda-forge wordcloud to download wordcloud. You will be asked whether you want to proceed or not. Type y for yes. Once that is done, type conda install -c conda-forge textblob to download textblob and y to proceed, and type conda install -c conda-forge gensim to download gensim and y to proceed.

Mac/Linux: Your terminal should already be open. Type command-t to open a new tab. Type conda install -c conda-forge wordcloud to download wordcloud. You will be asked whether you want to proceed or not. Type y for yes. Once that is done, type conda install -c conda-forge textblob to download textblob and y to proceed, and type conda install -c conda-forge gensim to download gensim and y to proceed.

If you have any issues, please email me at winstonjyoung@gmail.com.
