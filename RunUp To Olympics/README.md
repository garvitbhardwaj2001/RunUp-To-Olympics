# Olympics: Data Mining Project
Perform exploratory data mining and analysis on data available from Olympics and Asian games and study trends, patterns and factors that justify the medal tally of participating nations.

## DATA (from Kaggle, World Bank, Wikipedia and other websites):
Athlete_events.csv: Contains data of every participant that participated in Olympics from 1896 to 2016. Unzip the data.zip file provided.
* https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results

Population.csv:  Contains year-wise population data for 200+ countries from 1960 to 2020.
* https://data.worldbank.org/indicator/SP.POP.TOTL

Gdp_per_capita.csv: Contains year-wise gdp per capita for 200+ countries in US$ from 1960 to 2020.
* https://data.worldbank.org/indicator/NY.GDP.PCAP.CD

Calorie-intake.csv: Contains year-wise per capita calorie consumption data for various countries.
* https://ourworldindata.org/diet-compositions#diet-compositions-by-macronutrient

## DATA (We created):
medalsTally.csv: medal tally of each country that participated in Olympics from 1896 to 2020, scrapped this data from wikipedia. Can be found in data folder in the project’s directory.
asianGames.csv:  Contains the medal tally for each participating nation in Asian Games of 2018, scraped the data from sportskeeda and liquipedia websites. Can be found in the data folder as well.

## What we did?
* Data Cleansing and preprocessing
* Data Creation
* Data Merging
* Data Visualization and Inferencing
* Linear and Polynomial Regression
* K-Means Clustering
* Association Rule Mining

## What there is into this repository?
* Datasets we used.
* code to clean, transform and prepare for the analysis
* Some pictures and json files with charts/statistics from the data.
* A report and a presentation with detailed steps for all parts of the assignment.

## How to run the code?
* preprocess.py: File to convert and clean Table-1.csv(obtained by scraping data from wikipedia) to generate MedalsTally.csv. 
<br />**Input**: data/Table-1.csv
<br />**Output**: data/medalsTally.csv

* calAndGdp.py: File to perform the analysis of calories vs population and medals vs gdp-per-capita
<br />**Input**: data/medalsTally.csv, data/calorie-intake.csv, data/gdp_per_capita.csv, data/population.csv, data/AsianGames.py
<br />**Output**: 
<br />plot/Importance-Of-Calories.jpg: plot for comparative study of population vs calories for medals in Olympics
<br />plot/Importance-Of-GDP-per-capita.jpg: plot for comparative study of medal per million people vs gdp-per-capita for 38 participating nations in Asian Games 2018

* genderRatioYearWise.py: File to generate females to males ratio for Olympic games from 1896 to 2016 and generate a plot and perform Regression
<br />**Input**: data/athlete_events.csv
<br />**Output**:
<br />plots/genderRatio.jpg: scatter plot of gender ratio(females/males) plotted for olympics vs year
<br />plots/regressionOnGenderRatio: previous scatter plot fitted with a polynomial based regression model of degree-4

* clustering_ht_wt.py: File to generate the clusters based on height and weight of participants in Rio Olympics 2016
<br />**Input**: data/athlete_events.csv
<br />**Output**: 
<br />plots/heigh-weight-clusters.jpg: Plot containing 6 clusters of participants based on their height and weight using K-means clustering
<br />json/clusterWiseGames.json: Dictionary containing top 3 games with highest percentage of participation in each cluster 

* exploration.py: File to generate plots for historically most dominating countries at Olympics and for sports with most gold medals for participants aged above 50 along with few other experiments for exploring the datafiles
<br />**Input**: data/athlete_events.csv, data/noc_regions.csv
<br />**Output**: 
<br />plots/most_dominant_countries.jpg
<br />plots/medals_above_50.jpg

## What modules did we use?:
numpy, sklearn, scipy, pandas, matplotlib, seaborn, re
