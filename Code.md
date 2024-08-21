


# A. DOMAIN-LEVEL DATA ANALYSIS (All domain-level data were analyzed via Python, based on the following codes through 15 steps)



## 1. Access data and install packages
```
### Import packages for data access and analysis
### Meta-specific libraries
from fbri.private.sql.query import execute
from svinfer.linear_model import LinearRegression
from svinfer.processor import DataFrameProcessor
from svinfer.summary_statistics import SummaryStatistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns

### Accessible Meta database and tables
database = "fbri_prod_private"
attributes_table = "erc_condor_url_attributes_dp_final_v3"
breakdowns_table = "erc_condor_url_breakdowns_dp_clean_partitioned_v2"

```
## 2. Retrieve top-shared 5000 URLs with political_page_affinity (user political affinity group) = -2
```
### Define query criteria; limit the country to the US
target_domain = f"""
WITH RANK AS 
(SELECT parent_domain, COUNT(parent_domain) as frequency 
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
GROUP BY parent_domain
ORDER by frequency DESC
LIMIT 5000)
SELECT parent_domain
FROM RANK
"""

sql = f"""
WITH aggregated_urlbds AS
    (SELECT 
        urlbd.url_rid AS url_rid,
        min(urlbd.political_page_affinity) as political_page_affinity,
        min(urlbd.year_month) as year_month,
        SUM(urlbd.clicks) AS clicks,
        SUM(urlbd.shares) AS shares,
        SUM(urlbd.share_without_clicks) AS share_without_clicks
    FROM {database}.{breakdowns_table} urlbd
    WHERE urlbd.c='US' and urlbd.year_month <= '2020-12' and urlbd.political_page_affinity = '-2'
    GROUP BY urlbd.url_rid
    )
SELECT
    urlattr.url_rid,
    urlattr.parent_domain,
    aggregated_urlbds.clicks,
    aggregated_urlbds.shares,
    aggregated_urlbds.share_without_clicks,
    aggregated_urlbds.political_page_affinity
FROM
    {database}.{attributes_table} urlattr 
    JOIN aggregated_urlbds on urlattr.url_rid = aggregated_urlbds.url_rid
    WHERE urlattr.public_shares_top_country='US' AND LOWER(urlattr.parent_domain) IN ({target_domain})
"""
dfneg2 = execute(sql)
```

## 3. Retrieve top-shared 5000 URLs with political_page_affinity = -1
```
target_domain = f"""
WITH RANK AS 
(SELECT parent_domain, COUNT(parent_domain) as frequency 
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
GROUP BY parent_domain
ORDER by frequency DESC
LIMIT 5000)
SELECT parent_domain
FROM RANK
"""

sql = f"""
WITH aggregated_urlbds AS
    (SELECT 
        urlbd.url_rid AS url_rid,
        min(urlbd.political_page_affinity) as political_page_affinity,
        min(urlbd.year_month) as year_month,
        SUM(urlbd.clicks) AS clicks,
        SUM(urlbd.shares) AS shares,
        SUM(urlbd.share_without_clicks) AS share_without_clicks
    FROM {database}.{breakdowns_table} urlbd
    WHERE urlbd.c='US' and urlbd.year_month <= '2020-12' and urlbd.political_page_affinity = '-1'
    GROUP BY urlbd.url_rid
    )
SELECT
    urlattr.url_rid,
    urlattr.parent_domain,
    urlattr.public_shares_top_country,
    aggregated_urlbds.clicks,
    aggregated_urlbds.shares,
    aggregated_urlbds.share_without_clicks,
    aggregated_urlbds.political_page_affinity
FROM
    {database}.{attributes_table} urlattr 
    JOIN aggregated_urlbds on urlattr.url_rid = aggregated_urlbds.url_rid
    WHERE urlattr.public_shares_top_country='US' AND LOWER(urlattr.parent_domain) IN ({target_domain})
"""
dfneg1 = execute(sql)
```

## 4. Retrieve top-shared 5000 URLs with political_page_affinity = 0
```
target_domain = f"""
WITH RANK AS 
(SELECT parent_domain, COUNT(parent_domain) as frequency 
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
GROUP BY parent_domain
ORDER by frequency DESC
LIMIT 5000)
SELECT parent_domain
FROM RANK
"""

sql = f"""
WITH aggregated_urlbds AS
    (SELECT 
        urlbd.url_rid AS url_rid,
        min(urlbd.political_page_affinity) as political_page_affinity,
        min(urlbd.year_month) as year_month,
        SUM(urlbd.clicks) AS clicks,
        SUM(urlbd.shares) AS shares,
        SUM(urlbd.share_without_clicks) AS share_without_clicks
    FROM {database}.{breakdowns_table} urlbd
    WHERE urlbd.c='US' and urlbd.year_month <= '2020-12' and urlbd.political_page_affinity = '0'
    GROUP BY urlbd.url_rid
    )
SELECT
    urlattr.url_rid,
    urlattr.parent_domain,
    urlattr.public_shares_top_country,
    aggregated_urlbds.clicks,
    aggregated_urlbds.shares,
    aggregated_urlbds.share_without_clicks,
    aggregated_urlbds.political_page_affinity
FROM
    {database}.{attributes_table} urlattr 
    JOIN aggregated_urlbds on urlattr.url_rid = aggregated_urlbds.url_rid
    WHERE urlattr.public_shares_top_country='US' AND LOWER(urlattr.parent_domain) IN ({target_domain})
"""
dfzero = execute(sql)
```


## 5. Retrieve top-shared 5000 URLs with political_page_affinity = +1
```
target_domain = f"""
WITH RANK AS 
(SELECT parent_domain, COUNT(parent_domain) as frequency 
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
GROUP BY parent_domain
ORDER by frequency DESC
LIMIT 5000)
SELECT parent_domain
FROM RANK
"""

sql = f"""
WITH aggregated_urlbds AS
    (SELECT 
        urlbd.url_rid AS url_rid,
        min(urlbd.political_page_affinity) as political_page_affinity,
        min(urlbd.year_month) as year_month,
        SUM(urlbd.clicks) AS clicks,
        SUM(urlbd.shares) AS shares,
        SUM(urlbd.share_without_clicks) AS share_without_clicks
    FROM {database}.{breakdowns_table} urlbd
    WHERE urlbd.c='US' and urlbd.year_month <= '2020-12' and urlbd.political_page_affinity = '1'
    GROUP BY urlbd.url_rid
    )
SELECT
    urlattr.url_rid,
    urlattr.parent_domain,
    urlattr.public_shares_top_country,
    aggregated_urlbds.clicks,
    aggregated_urlbds.shares,
    aggregated_urlbds.share_without_clicks,
    aggregated_urlbds.political_page_affinity
FROM
    {database}.{attributes_table} urlattr 
    JOIN aggregated_urlbds on urlattr.url_rid = aggregated_urlbds.url_rid
    WHERE urlattr.public_shares_top_country='US' AND LOWER(urlattr.parent_domain) IN ({target_domain})
"""
dfpos1 = execute(sql)
```

## 6. Retrieve top-shared 5000 URLs with political_page_affinity = +2
```
target_domain = f"""
WITH RANK AS 
(SELECT parent_domain, COUNT(parent_domain) as frequency 
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
GROUP BY parent_domain
ORDER by frequency DESC
LIMIT 5000)
SELECT parent_domain
FROM RANK
"""

sql = f"""
WITH aggregated_urlbds AS
    (SELECT 
        urlbd.url_rid AS url_rid,
        min(urlbd.political_page_affinity) as political_page_affinity,
        min(urlbd.year_month) as year_month,
        SUM(urlbd.clicks) AS clicks,
        SUM(urlbd.shares) AS shares,
        SUM(urlbd.share_without_clicks) AS share_without_clicks
    FROM {database}.{breakdowns_table} urlbd
    WHERE urlbd.c='US' and urlbd.year_month <= '2020-12' and urlbd.political_page_affinity = '2'
    GROUP BY urlbd.url_rid
    )
SELECT
    urlattr.url_rid,
    urlattr.parent_domain,
    urlattr.public_shares_top_country,
    aggregated_urlbds.clicks,
    aggregated_urlbds.shares,
    aggregated_urlbds.share_without_clicks,
    aggregated_urlbds.political_page_affinity
FROM
    {database}.{attributes_table} urlattr 
    JOIN aggregated_urlbds on urlattr.url_rid = aggregated_urlbds.url_rid
    WHERE urlattr.public_shares_top_country='US' AND LOWER(urlattr.parent_domain) IN ({target_domain})
"""
dfpos2 = execute(sql)
```

## 7. Combine all 5 political_page_affinity groups
```
### Use Pandas dataframe for the operation
frames = [dfneg2, dfneg1, dfzero, dfpos1, dfpos2]
df_all = pd.concat(frames)
```

## 8. Group df_all by parent domain while retaining political affinity values
```
df_parent = pd.DataFrame({'count_url' : df_all.groupby(['parent_domain','political_page_affinity']).size(), 'total_clicks': df_all.groupby(['parent_domain','political_page_affinity'])['clicks'].sum(), 'total_shares': df_all.groupby(['parent_domain','political_page_affinity'])['shares'].sum(), 'total_sharewoclicks': df_all.groupby(['parent_domain','political_page_affinity'])['share_without_clicks'].sum()}).reset_index()
```

## 9. Calculate total_shares for each domain and append it to df_parent
```
df_shares = df_parent.groupby('parent_domain')['total_shares']
df_parent['sum_shares']=df_shares.transform('sum')
```

## 10. Calculate weighted affinity for each domain
```
weighted_affinity = df_parent.groupby('parent_domain').apply(lambda x: (x['political_page_affinity']*(x['total_shares']/x['sum_shares'])).sum()).reset_index()
weighted_affinity.columns = ['parent_domain', 'weighted_affinity']
```

## 11. Save data for analysis, which has 25000 rows
```
result_affinity = pd.merge(df_parent, weighted_affinity, on='parent_domain')
result_affinity["needed_shares"] = 1.578 * ((15805440*result_affinity["count_url"])**(0.5))
result_affinity["extra_shares"] = result_affinity["sum_shares"] - result_affinity["needed_shares"]
result_affinity.to_csv(r'top5000_byaffinity.csv')
```

## 12. Exclude domains lacking sufficient shares, missing political page affinity, and an outlier
```
df = pd.read_csv('top5000_byaffinity.csv')
df = df[df['extra_shares'] > 0]
df['affinity_alignment'] = abs(df['political_page_affinity']-df['weighted_affinity'])
df = df[df['parent_domain'] != 'https://urldefense.com/v3/__http://care-net.org__;!!DLa72PTfQgg!MLe3jtaenb_GC6RMaN3bUo_O892Xch2Ko2ASCTjE6eKc6g5uPxPNiyJSrQkw0TvHZuN_4NjN0MS5oQs4xdU$ ']
df = df[df['parent_domain'] != 'https://urldefense.com/v3/__http://osu.edu__;!!DLa72PTfQgg!MLe3jtaenb_GC6RMaN3bUo_O892Xch2Ko2ASCTjE6eKc6g5uPxPNiyJSrQkw0TvHZuN_4NjN0MS5BKvzLrc$ ']
df = df[df['parent_domain'] != 'https://urldefense.com/v3/__http://proconservativenews.com__;!!DLa72PTfQgg!MLe3jtaenb_GC6RMaN3bUo_O892Xch2Ko2ASCTjE6eKc6g5uPxPNiyJSrQkw0TvHZuN_4NjN0MS5hKE7waE$ ']
df = df[df['parent_domain'] != 'https://urldefense.com/v3/__http://whicdn.com__;!!DLa72PTfQgg!MLe3jtaenb_GC6RMaN3bUo_O892Xch2Ko2ASCTjE6eKc6g5uPxPNiyJSrQkw0TvHZuN_4NjN0MS553GjcqM$ ']
df = df[df['parent_domain'] != 'https://urldefense.com/v3/__http://youtube.com__;!!DLa72PTfQgg!MLe3jtaenb_GC6RMaN3bUo_O892Xch2Ko2ASCTjE6eKc6g5uPxPNiyJSrQkw0TvHZuN_4NjN0MS5RI8yDfE$ ']
```

## 13. Analyze affinity alignment effects on SwoCs
```
x_columns = ["affinity_alignment"]
y_column = "total_sharewoclicks"
x_s2 = [0]
a = sns.scatterplot(data=df, x="affinity_alignment", y="total_sharewoclicks")
a.set_xlabel("Affinity Alignment")
a.set_ylabel("Shares without Clicks")
df_data = DataFrameProcessor(df)
model2 = LinearRegression(x_columns, y_column, x_s2, random_state=123).fit(df_data)
print(f"beta_tilde is: \n{model2.beta}")
print(f"beta_tilde's standard error is: \n{model2.beta_standarderror}")
print(f"beta_tilde's variance-covariance matrix: \n{model2.beta_vcov}")
print(f"beta_tilde's residual variance is: \n{model2.sigma_sq}")
```

## 14. Return to step #11 and group 5 political affinity groups
```
df = pd.DataFrame({'count_domain': df.groupby('parent_domain').size(),
                   'weighted_affinity': df.groupby('parent_domain')['weighted_affinity'].first(),
                   'clicks': df.groupby('parent_domain')['total_clicks'].sum(),
                   'shares': df.groupby('parent_domain')['total_shares'].sum(),
                   'share_without_clicks': df.groupby('parent_domain')['total_sharewoclicks'].sum(),
                  }).reset_index()
df["weighted_affinity2"] = df["weighted_affinity"] * df["weighted_affinity"]
```

## 15. Analyze weighted content affinity effects on SwoCs
```
x_columns = ["weighted_affinity", "weighted_affinity2"]
y_column = "share_without_clicks"
x_s2 = [0, 0]
a = sns.scatterplot(data=df, x="weighted_affinity", y="share_without_clicks")
a.set_xlabel("Content Affinity")
a.set_ylabel("Shares without Clicks")
df_data = DataFrameProcessor(df)
model = LinearRegression(x_columns, y_column, x_s2, random_state=123).fit(df_data)
print(f"beta_tilde is: \n{model.beta}")
print(f"beta_tilde's standard error is: \n{model.beta_standarderror}")
print(f"beta_tilde's variance-covariance matrix: \n{model.beta_vcov}")
print(f"beta_tilde's residual variance is: \n{model.sigma_sq}")
```


# B1. POLITICAL CLASSIFICATION OF URLS (All URLs data are classified as political or non-political content for each year.)

## 1. Access data and install packages
```
### Import packages for data access and analysis
### Meta-specific libraries
from fbri.private.sql.query import execute

import pandas as pd
import csv

### Accessible Meta database and tables
database = "fbri_prod_private"
attributes_table = "erc_condor_url_attributes_dp_final_v3"
breakdowns_table = "erc_condor_url_breakdowns_dp_clean_partitioned_v2"
```

## 2. Retrieve text data from May to November URLs for each year (2017, 2018, 2019, and 2020, run separately iterated)
```
### Define query criteria and specify the year
sql = f"""
WITH May_Nov_US_URLs AS (
SELECT url_rid, share_title, share_main_blurb
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
)
SELECT
    min(May_Nov_US_URLs.share_title) as share_title,
    min(May_Nov_US_URLs.share_main_blurb) as share_main_blurb,
    urlbd.url_rid AS url_rid,
    SUM(urlbd.share_without_clicks) AS SwoC
FROM {database}.{breakdowns_table} urlbd
JOIN May_Nov_US_URLs ON
urlbd.url_rid = May_Nov_US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.year_month > '2017-04' and urlbd.year_month < '2017-12'
GROUP BY urlbd.url_rid
"""

### Save data to file as tab-separated values (TSV)
df_blurbs = execute(sql)
df_blurbs.to_csv('2017_blurbs_clean.tsv', sep='\t', index=False)
```

## 3. Prepare/process the combination of URL titles and blurbs into stems 
```
### Import Natural Language Toolkit (NLTK) libraries for text processing
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer

import string
import re
import pandas as pd

### Function for text preprocessing:
### Text cleaning by removing stop words, digits, and non-ASCII characters 
### Keep stems
def stem_prerpare(pathO, pathD):
    print("program starts ...")
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    mFile = open(pathO,'r',encoding='utf-8')
    oFile = open(pathD,'w',encoding='utf-8')
    next(mFile)
    for mRow in mFile:
        data = mRow.rstrip().split('\t')
        title = data[0]
        content = data[1]
        text = title.rstrip() + " " + content.rstrip()
        text = text.lower().split()
        text = [w for w in text if not w in stop and len(w) >= 3]
        text = [ch for ch in text if ch not in exclude]
        text = " ".join(text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"what\'s", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        mString = '\t'.join(data[2:]) + '\t' + text + '\n'
        oFile.write(mString)
    oFile.close()
    mFile.close()

### Save cleaned data to a file
stem_prerpare('2017_blurbs_clean.tsv', '2017_blurbs_clean_stem.tsv')
```

## 4. Classifier for political and non-political content classification
```
import sys
import string
import re

### Utilize scikit-learn libraries
### Add additional libraries for performance evaluation and visualization
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import numpy, string
import pandas

from matplotlib import pyplot as plt
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

### Function for training the model for the classifier
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return [metrics.recall_score(valid_y, predictions, average='weighted'), metrics.accuracy_score(valid_y, predictions), metrics.f1_score(valid_y, predictions, average='weighted')]

### Apply the classifier to the input file
###file1 is the input file
###file2 is the outfile with extra column of political class (0:non-political; 1:political)
def classify(file1, file2):
	print("program starts")
	mFile = open('./2017_Pol_Coding_processed.csv','r', encoding='utf-8')
	labels, texts = [], []
	for mRow in mFile:
		content = mRow.rstrip().split(',')
		labels.append(content[0].lower())
		texts.append(" ".join(content[1:]))
	
	trainDF = pandas.DataFrame()
	trainDF['text'] = texts
	trainDF['label'] = labels
	train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.2, shuffle=True, random_state=0)

	count_vect = CountVectorizer()
	tokenizer = TreebankWordTokenizer()
	count_vect.set_params(tokenizer=tokenizer.tokenize)
	count_vect.set_params(stop_words='english')
	count_vect.set_params(ngram_range=(1,2))
	count_vect.set_params(max_df=0.5)
	count_vect.set_params(min_df=2)

	count_vect.fit(trainDF['text'])

	xtrain_count =  count_vect.transform(train_x)
	xvalid_count =  count_vect.transform(valid_x)

	encoder = preprocessing.LabelEncoder()
	train_y = encoder.fit_transform(train_y)
	valid_y = encoder.fit_transform(valid_y)

	### Print the information for accuracy
	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count,valid_y)
	print("Random Forest, Count Vectors recall: ", accuracy[0])
	print("Random Forest, Count Vectors precision: ", accuracy[1])
	print("Random Forest, Count Vectors F1 score: ", accuracy[2])

	clf = ensemble.RandomForestClassifier(random_state=0).fit(xtrain_count,train_y)
	
	pFile = open(file1,'r',encoding='utf-8')
	oFile = open(file2,'w',encoding='utf-8')
	
	for mRow in pFile:
		texts = []
		content = mRow.strip().split("\t")
		texts.append(" ".join(content[-1].split()))
		xx = count_vect.transform(texts)
		results = clf.predict(xx)
		mString = mRow.strip() + "\t" + str(results[0]) + "\n"
		oFile.write(mString)
	oFile.close()

### Perform classification and save the data to a new file
classify('./2017_blurbs_clean_stem.tsv', './2017MayNov_pol.tsv')
```

## 5. Read df_pol file with URLs politically classified
```
df_pol = pd.read_csv(r'2017MayNov_pol.tsv', delimiter = '\t', header = None, names = ['url_rid','SwoC','titleNblurb','political'])
```


# B2. URL-LEVEL DATA PREPARATION AND MAJOR ANALYSIS (All URL-level data were analyzed using Python through the following 19 steps)

## 1. Access data and install packages
```
### Import packages for data access and analysis
### Meta-specific libraries
from fbri.private.sql.query import execute

### Libraries for dealing with differential privacy
from svinfer.linear_model import LinearRegression
from svinfer.processor import DataFrameProcessor
from svinfer.summary_statistics import SummaryStatistics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


import seaborn as sns

### Accessible Meta database and tables
database = "fbri_prod_private"
attributes_table = "erc_condor_url_attributes_dp_final_v3"
breakdowns_table = "erc_condor_url_breakdowns_dp_clean_partitioned_v2"
```

## 2. Select data from May to Nov with political_page_affinity (user political affinity group) = -2 for each year (2017, 2018, 2019, 2020 run separately iterated)
```
sql = f"""
WITH May_Nov_US_URLs AS (
SELECT url_rid, parent_domain
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
)
SELECT 
    urlbd.url_rid AS url_rid,
    min(urlbd.political_page_affinity) as political_page_affinity,
    min(urlbd.year_month) as year_month,
    SUM(urlbd.clicks) AS clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.share_without_clicks) AS share_without_clicks
FROM {database}.{breakdowns_table} urlbd
JOIN May_Nov_US_URLs ON
urlbd.url_rid = May_Nov_US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.year_month > '2017-04' and urlbd.year_month < '2017-12' and urlbd.political_page_affinity = '-2' 
GROUP BY urlbd.url_rid
"""
dfneg2 = execute(sql)
```

## Select data from May to Nov with political_page_affinity (user political affinity group) = -1 for each year (2017, 2018, 2019, 2020 run separately iterated)
```
sql = f"""
WITH May_Nov_US_URLs AS (
SELECT url_rid, parent_domain
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
)
SELECT 
    urlbd.url_rid AS url_rid,
    min(urlbd.political_page_affinity) as political_page_affinity,
    min(urlbd.year_month) as year_month,
    SUM(urlbd.clicks) AS clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.share_without_clicks) AS share_without_clicks
FROM {database}.{breakdowns_table} urlbd
JOIN May_Nov_US_URLs ON
urlbd.url_rid = May_Nov_US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.year_month > '2017-04' and urlbd.year_month < '2017-12' and urlbd.political_page_affinity = '-1' 
GROUP BY urlbd.url_rid
"""
dfneg1 = execute(sql)
```

## 4. Select data from May to Nov with political_page_affinity (user political affinity group) = 0 for each year (2017, 2018, 2019, 2020 run separately iterated)
```
sql = f"""
WITH May_Nov_US_URLs AS (
SELECT url_rid, parent_domain
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
)
SELECT 
    urlbd.url_rid AS url_rid,
    min(urlbd.political_page_affinity) as political_page_affinity,
    min(urlbd.year_month) as year_month,
    SUM(urlbd.clicks) AS clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.share_without_clicks) AS share_without_clicks
FROM {database}.{breakdowns_table} urlbd
JOIN May_Nov_US_URLs ON
urlbd.url_rid = May_Nov_US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.year_month > '2017-04' and urlbd.year_month < '2017-12' and urlbd.political_page_affinity = '0' 
GROUP BY urlbd.url_rid
"""
dfzero = execute(sql)
```

## 5. Select data from May to Nov with political_page_affinity (user political affinity group) = +1 for each year (2017, 2018, 2019, 2020 run separately iterated)
```
sql = f"""
WITH May_Nov_US_URLs AS (
SELECT url_rid, parent_domain
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
)
SELECT 
    urlbd.url_rid AS url_rid,
    min(urlbd.political_page_affinity) as political_page_affinity,
    min(urlbd.year_month) as year_month,
    SUM(urlbd.clicks) AS clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.share_without_clicks) AS share_without_clicks
FROM {database}.{breakdowns_table} urlbd
JOIN May_Nov_US_URLs ON
urlbd.url_rid = May_Nov_US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.year_month > '2017-04' and urlbd.year_month < '2017-12' and urlbd.political_page_affinity = '1' 
GROUP BY urlbd.url_rid
"""
dfpos1 = execute(sql)
```

## 6. Select data from May to Nov with political_page_affinity (user political affinity group) = +2 for each year (2017, 2018, 2019, 2020 run separately iterated)
```
sql = f"""
WITH May_Nov_US_URLs AS (
SELECT url_rid, parent_domain
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US'
)
SELECT 
    urlbd.url_rid AS url_rid,
    min(urlbd.political_page_affinity) as political_page_affinity,
    min(urlbd.year_month) as year_month,
    SUM(urlbd.clicks) AS clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.share_without_clicks) AS share_without_clicks
FROM {database}.{breakdowns_table} urlbd
JOIN May_Nov_US_URLs ON
urlbd.url_rid = May_Nov_US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.year_month > '2017-04' and urlbd.year_month < '2017-12' and urlbd.political_page_affinity = '2' 
GROUP BY urlbd.url_rid
"""
dfpos2 = execute(sql)
```

## 7. Combine all URLs from 5 user political affinities
```
frames = [dfneg2, dfneg1, dfzero, dfpos1, dfpos2]
df_all = pd.concat(frames)
```

## 8. Calculate sum_shares for each URL to obtain the weighted political affinity
```
df_shares = df_all.groupby('url_rid')['shares']
df_all['sum_shares']=df_shares.transform('sum')
```

## 9. Group by URL and calculate weighted political affinity per URL
```
df_url = pd.DataFrame({'clicks': df_all.groupby('url_rid')['clicks'].sum(),
                   'shares': df_all.groupby('url_rid')['shares'].sum(),
                   'share_without_clicks': df_all.groupby('url_rid')['share_without_clicks'].sum(),
                  }).reset_index()
weighted_affinity = df_all.groupby('url_rid').apply(lambda x: (x['political_page_affinity']*(x['shares']/x['sum_shares'])).sum()).reset_index()
weighted_affinity.columns = ['url_rid', 'weighted_affinity']
```

## 10. Create a dataset with URL aggregates and weighted affinity combined
```
result = pd.merge(df_url, weighted_affinity, on='url_rid')
```

## 11. Read the classified URL file and merge it with the result
```
df_pol = pd.read_csv(r'2017MayNov_pol.tsv', delimiter = '\t', header = None, names = ['url_rid','SwoC','titleNblurb','political'])
df_combined = pd.merge(result, df_pol, on='url_rid')
```

## 12. Exclude URLs with shares below 2395 (not meeting size requirements for noise adjustment)
```
df_combined_filtered = df_combined[df_combined['shares'] >2395]
```

## 13. Create a dataset divided by 5 affinity levels
```
result_affinity = pd.merge(df_all, weighted_affinity, on='url_rid')
```

## 14. Read the classified URL file and merge it with df_all with all user affinity
```
df_combined_affinity = pd.merge(result_affinity, df_pol, on='url_rid')
```

## 15. Exclude URLs with total shares below 2395 (not meeting size requirements for noise adjustment)s
```
df_combined_affinity_filtered = df_combined_affinity[df_combined_affinity['sum_shares'] > 2395
```

## 16. Check the URL count for each political classifier
```
df = df_combined_filtered
df = pd.DataFrame({'count' : df.groupby('political').size(), 
                  }).reset_index()
```

## 17. Get each user affinity aggregates for political content (adjust political == 0 for non-political content)
```
from numpy import sqrt
from numpy import power

df = df_combined_affinity
df = df[df['political'] == 1]
 
df = pd.DataFrame({'count': df.groupby('political_page_affinity').size(),
                   'share_without_clicks': df.groupby('political_page_affinity')['share_without_clicks'].sum(),
                   'shares': df.groupby('political_page_affinity')['shares'].sum(),
                   'clicks': df.groupby('political_page_affinity')['clicks'].sum(),
             	 }).reset_index()
 
df['SwoC_ULCI']=round(df['share_without_clicks']+1.96*sqrt(df['count']*power(10,2)))
df['SwoC_LLCI']=round(df['share_without_clicks']-1.96*sqrt(df['count']*power(10,2)))
df['Shares_ULCI']=round(df['shares']+1.96*sqrt(df['count']*power(14,2)))
df['Shares_LLCI']=round(df['shares']-1.96*sqrt(df['count']*power(14,2)))
df['Clicks_ULCI']=round(df['clicks']+1.96*sqrt(df['count']*power(40,2)))
df['Clicks_LLCI']=round(df['clicks']-1.96*sqrt(df['count']*power(40,2)))
 
df['SwoC/count'] = round(df['share_without_clicks']/df['count'])
df['shares/count'] = round(df['shares']/df['count'])
df['clicks/count'] = round(df['clicks']/df['count'])
df['percentage'] = round(df['share_without_clicks']/df['shares']*100,2)
```

## 18. Calculate a) weighted affinity^2, b) filter -2 < weighted affinity <2, and c) only include political URLs to run quadratic regression model with x = weighted affinity, x2= weighted affinity^2, y = SwoC 
```
df = df_combined_filtered
df['weighted_affinity2'] = df['weighted_affinity']*df['weighted_affinity'] 
df = df[df['political'] == 1]
df = df[df['weighted_affinity'] <= 2]
df = df[df['weighted_affinity'] >= -2]
df = df.drop(labels=1330466, axis=0)
x_columns = ["weighted_affinity", "weighted_affinity2"]
y_column = "share_without_clicks"
x_s2 = [0, 0]
a = sns.scatterplot(data=df, x="weighted_affinity", y="share_without_clicks")
a.set_xlabel("Content Affinity")
a.set_ylabel("Shares without Clicks")
df_data = DataFrameProcessor(df)
model = LinearRegression(x_columns, y_column, x_s2, random_state=123).fit(df_data)
print(f"beta_tilde is: \n{model.beta}")
print(f"beta_tilde's standard error is: \n{model.beta_standarderror}")
print(f"beta_tilde's variance-covariance matrix: \n{model.beta_vcov}")
print(f"beta_tilde's residual variance is: \n{model.sigma_sq}")
```

## 19. a) calculate “alignment”, b) filter -2 < weighted affinity <2, and c) only include political content to run a single linear regression model with x = alignment and y = SwoC
```
df = df_combined_affinity_filtered
df['alignment'] = abs(df['political_page_affinity']-df['weighted_affinity'])
df = df[df['political'] == 1]
df = df[df['weighted_affinity'] <= 2]
df = df[df['weighted_affinity'] >= -2]
df = df.drop(labels=[17064170, 17064171, 17064172, 17064173, 17064174], axis=0)
x_columns = ["alignment"]
y_column = "share_without_clicks"
x_s2 = [0]
a = sns.scatterplot(data=df, x="alignment", y="share_without_clicks")
a.set_xlabel("Affinity Alignment")
a.set_ylabel("Shares without Clicks")
df_data = DataFrameProcessor(df)
model2 = LinearRegression(x_columns, y_column, x_s2, random_state=123).fit(df_data)
print(f"beta_tilde is: \n{model2.beta}")
print(f"beta_tilde's standard error is: \n{model2.beta_standarderror}")
print(f"beta_tilde's variance-covariance matrix: \n{model2.beta_vcov}")
print(f"beta_tilde's residual variance is: \n{model2.sigma_sq}")
```

# C. TPFC-RATED URLS DATA PREPARATION AND MAJOR ANALYSIS (All TPFC-rated URLs were filtered, politically classified, and analyzed utilizing the Python codes below.)

## 1. Access data and install packages
```
### Import packages for data access and analysis
### Meta specific libraries
from fbri.private.sql.query import execute
from svinfer.linear_model import LinearRegression
from svinfer.processor import DataFrameProcessor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import seaborn as sns

###Accessible Meta database and tables
database = "fbri_prod_private"
attributes_table = "erc_condor_url_attributes_dp_final_v3"
breakdowns_table = "erc_condor_url_breakdowns_dp_clean_partitioned_v2"
```

## 2. Extract all tpfc_rated URLs
```
sql = f"""
WITH US_URLs AS (
SELECT url_rid, clean_url, parent_domain, tpfc_rating, tpfc_first_fact_check, share_title, share_main_blurb
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US' AND tpfc_rating != 'NaN'
)
SELECT
    min(US_URLs.share_title) as share_title,
    min(US_URLs.share_main_blurb) as share_main_blurb,
    urlbd.url_rid AS url_rid,
    min(US_URLs.clean_url) as clean_url,
    min(US_URLs.parent_domain) as parent_domain,
    min(US_URLs.tpfc_rating) as tpfc_rating,
    min(urlbd.year_month) as year_month,
    SUM(urlbd.share_without_clicks) AS share_without_clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.clicks) AS clicks
FROM {database}.{breakdowns_table} urlbd
JOIN US_URLs ON
urlbd.url_rid = US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.political_page_affinity IS NOT NULL
GROUP BY urlbd.url_rid
"""

tpfc_year = execute(sql)
```

## 3. Add year and month data and split each year
```
from datetime import datetime

tpfc_year['year']= pd.to_datetime(tpfc_year['year_month']).dt.to_period('Y')
tpfc_year.to_csv(r'tpfc_4yrs.csv')

df_2017 = tpfc_year[tpfc_year['year'] == '2017']
df_2017.to_csv(r'tpfc_2017.csv')

df_2018 = tpfc_year[tpfc_year['year'] == '2018']
df_2018.to_csv(r'tpfc_2018.csv')

df_2019 = tpfc_year[tpfc_year['year'] == '2019']
df_2019.to_csv(r'tpfc_2019.csv')

df_2020 = tpfc_year[tpfc_year['year'] == '2020']
df_2020.to_csv(r'tpfc_2020.csv')
```

## 4. Prepare/process URL titles and blurbs for political classification for each year (this is a 2017 example, run separately for other years)
```
df = pd.read_csv('tpfc_2017.csv', delimiter = ',', error_bad_lines=False)
df.to_csv('tpfc_2017_clean.tsv', sep='\t', index=False)

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import string
import re
import pandas as pd

## Function for text preprocessing
def stem_prerpare(pathO, pathD):
    print("program starts ...")
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer()
    mFile = open(pathO,'r',encoding='utf-8')
    oFile = open(pathD,'w',encoding='utf-8')
    next(mFile)
    for mRow in mFile:
        data = mRow.rstrip().split('\t')
        title = data[1]
        content = data[2]
        text = title.rstrip() + " " + content.rstrip()
        text = text.lower().split()
        text = [w for w in text if not w in stop and len(w) >= 3]
        text = [ch for ch in text if ch not in exclude]
        text = " ".join(text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"what\'s", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n\'t", " not ", text)
        text = re.sub(r"i\'m", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
        mString = '\t'.join(data[3:]) + '\t' + text + '\n'
        oFile.write(mString)
    oFile.close()
    mFile.close()

stem_prerpare('tpfc_2017_clean.tsv', 'tpfc_2017_clean_stem.tsv')
```

## 5. Perform political classification and save the classified URL files for each year (this is a 2017 example, run separately for other years)
```
import sys
import string
import re

### Utilize scikit-learn libraries 
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import numpy, string
import pandas

from matplotlib import pyplot as plt
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist

### Function to train the model
def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    classifier.fit(feature_vector_train, label)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    return [metrics.recall_score(valid_y, predictions, average='weighted'), metrics.accuracy_score(valid_y, predictions), metrics.f1_score(valid_y, predictions, average='weighted')]

### Apply the classifier to the input file
###file1 is the input file
###file2 is the outfile with extra column of political class (0:non-political; 1:political)
def classify(file1, file2):
	print("program starts")
	mFile = open('./2020_Pol_Coding_processed.csv','r', encoding='utf-8')
	labels, texts = [], []
	for mRow in mFile:
		content = mRow.rstrip().split(',')
		labels.append(content[0].lower())
		texts.append(" ".join(content[1:]))
	
	trainDF = pandas.DataFrame()
	trainDF['text'] = texts
	trainDF['label'] = labels
	train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=0.2, shuffle=True, random_state=0)

	count_vect = CountVectorizer()
	tokenizer = TreebankWordTokenizer()
	count_vect.set_params(tokenizer=tokenizer.tokenize)
	count_vect.set_params(stop_words='english')
	count_vect.set_params(ngram_range=(1,2))
	count_vect.set_params(max_df=0.5)
	count_vect.set_params(min_df=2)

	count_vect.fit(trainDF['text'])

	xtrain_count =  count_vect.transform(train_x)
	xvalid_count =  count_vect.transform(valid_x)

	encoder = preprocessing.LabelEncoder()
	train_y = encoder.fit_transform(train_y)
	valid_y = encoder.fit_transform(valid_y)

	accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count,valid_y)
	print("Random Forest, Count Vectors recall: ", accuracy[0])
	print("Random Forest, Count Vectors precision: ", accuracy[1])
	print("Random Forest, Count Vectors F1 score: ", accuracy[2])

	clf = ensemble.RandomForestClassifier(random_state=0).fit(xtrain_count,train_y)
	
	pFile = open(file1,'r',encoding='utf-8')
	oFile = open(file2,'w',encoding='utf-8')
	
	for mRow in pFile:
		texts = []
		content = mRow.strip().split("\t")
		texts.append(" ".join(content[-1].split()))
		xx = count_vect.transform(texts)
		results = clf.predict(xx)
		mString = mRow.strip() + "\t" + str(results[0]) + "\n"
		oFile.write(mString)
	oFile.close()

### Perform classification and save the results to a new file
classify('./tpfc_2017_clean_stem.tsv', './tpfc_2017_pol.tsv')
```

## 6. Repeat the above step for each year, and combine all tpfc-rated political files
```
df_2017 = pd.read_csv(r'tpfc_2017_pol.tsv', delimiter = '\t', error_bad_lines=False, header = None, names = ['url_rid','clean_url','parent_domain','tpfc_rating','year_month','share_without_clicks','shares','clicks','year','titleNblurb','political'])
df_2018 = pd.read_csv(r'tpfc_2018_pol.tsv', delimiter = '\t', error_bad_lines=False, header = None, names = ['url_rid','clean_url','parent_domain','tpfc_rating','year_month','share_without_clicks','shares','clicks','year','titleNblurb','political'])
df_2019 = pd.read_csv(r'tpfc_2019_pol.tsv', delimiter = '\t', error_bad_lines=False, header = None, names = ['url_rid','clean_url','parent_domain','tpfc_rating','year_month','share_without_clicks','shares','clicks','year','titleNblurb','political'])
df_2020 = pd.read_csv(r'tpfc_2020_pol.tsv', delimiter = '\t', error_bad_lines=False, header = None, names = ['url_rid','clean_url','parent_domain','tpfc_rating','year_month','share_without_clicks','shares','clicks','year','titleNblurb','political'])
frames = [df_2017, df_2018, df_2019, df_2020]

### Combine all outputs and save it to a new file
tpfc_pol = pd.concat(frames)
tpfc_pol.to_csv(r'tpfc_pol.csv')
```

## 7. Select tpfc-ratled URLs with political_page_affinity (user political affinity group) = -2 for each year (2017, 2018, 2019, 2020 run separately iterated)
```
sql = f"""
WITH US_URLs AS (
SELECT url_rid, clean_url, parent_domain, tpfc_rating, tpfc_first_fact_check, share_title, share_main_blurb
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US' AND tpfc_rating != 'NaN'
)
SELECT
    min(US_URLs.share_title) as share_title,
    min(US_URLs.share_main_blurb) as share_main_blurb,
    urlbd.url_rid AS url_rid,
    min(US_URLs.clean_url) as clean_url,
    min(US_URLs.parent_domain) as parent_domain,
    min(US_URLs.tpfc_rating) as tpfc_rating,
    min(US_URLs.tpfc_first_fact_check) as tpfc_first_fact_check,
    min(urlbd.year_month) as year_month,
    min(urlbd.political_page_affinity) as political_page_affinity,
    SUM(urlbd.share_without_clicks) AS share_without_clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.clicks) AS clicks
FROM {database}.{breakdowns_table} urlbd
JOIN US_URLs ON
urlbd.url_rid = US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.political_page_affinity = '-2'
GROUP BY urlbd.url_rid
"""

dfneg2 = execute(sql)
```

## 8. Select tpfc-ratled URLs with political_page_affinity = -1
```
sql = f"""
WITH US_URLs AS (
SELECT url_rid, clean_url, parent_domain, tpfc_rating, tpfc_first_fact_check, share_title, share_main_blurb
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US' AND tpfc_rating != 'NaN'
)
SELECT
    min(US_URLs.share_title) as share_title,
    min(US_URLs.share_main_blurb) as share_main_blurb,
    urlbd.url_rid AS url_rid,
    min(US_URLs.clean_url) as clean_url,
    min(US_URLs.parent_domain) as parent_domain,
    min(US_URLs.tpfc_rating) as tpfc_rating,
    min(US_URLs.tpfc_first_fact_check) as tpfc_first_fact_check,
    min(urlbd.year_month) as year_month,
    min(urlbd.political_page_affinity) as political_page_affinity,
    SUM(urlbd.share_without_clicks) AS share_without_clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.clicks) AS clicks
FROM {database}.{breakdowns_table} urlbd
JOIN US_URLs ON
urlbd.url_rid = US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.political_page_affinity = '-1'
GROUP BY urlbd.url_rid
"""

dfneg1 = execute(sql)
```

## 9. Select tpfc-ratled URLs with political_page_affinity = 0
```
sql = f"""
WITH US_URLs AS (
SELECT url_rid, clean_url, parent_domain, tpfc_rating, tpfc_first_fact_check, share_title, share_main_blurb
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US' AND tpfc_rating != 'NaN'
)
SELECT
    min(US_URLs.share_title) as share_title,
    min(US_URLs.share_main_blurb) as share_main_blurb,
    urlbd.url_rid AS url_rid,
    min(US_URLs.clean_url) as clean_url,
    min(US_URLs.parent_domain) as parent_domain,
    min(US_URLs.tpfc_rating) as tpfc_rating,
    min(US_URLs.tpfc_first_fact_check) as tpfc_first_fact_check,
    min(urlbd.year_month) as year_month,
    min(urlbd.political_page_affinity) as political_page_affinity,
    SUM(urlbd.share_without_clicks) AS share_without_clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.clicks) AS clicks
FROM {database}.{breakdowns_table} urlbd
JOIN US_URLs ON
urlbd.url_rid = US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.political_page_affinity = '0'
GROUP BY urlbd.url_rid
"""

dfzero = execute(sql)
```

## 10. Select tpfc-ratled URLs with political_page_affinity = +1
```
sql = f"""
WITH US_URLs AS (
SELECT url_rid, clean_url, parent_domain, tpfc_rating, tpfc_first_fact_check, share_title, share_main_blurb
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US' AND tpfc_rating != 'NaN'
)
SELECT
    min(US_URLs.share_title) as share_title,
    min(US_URLs.share_main_blurb) as share_main_blurb,
    urlbd.url_rid AS url_rid,
    min(US_URLs.clean_url) as clean_url,
    min(US_URLs.parent_domain) as parent_domain,
    min(US_URLs.tpfc_rating) as tpfc_rating,
    min(US_URLs.tpfc_first_fact_check) as tpfc_first_fact_check,
    min(urlbd.year_month) as year_month,
    min(urlbd.political_page_affinity) as political_page_affinity,
    SUM(urlbd.share_without_clicks) AS share_without_clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.clicks) AS clicks
FROM {database}.{breakdowns_table} urlbd
JOIN US_URLs ON
urlbd.url_rid = US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.political_page_affinity = '1'
GROUP BY urlbd.url_rid
"""

dfpos1 = execute(sql)
```

## 11. Select tpfc-ratled URLs with political_page_affinity = +2
```
sql = f"""
WITH US_URLs AS (
SELECT url_rid, clean_url, parent_domain, tpfc_rating, tpfc_first_fact_check, share_title, share_main_blurb
FROM {database}.{attributes_table}
WHERE public_shares_top_country = 'US' AND tpfc_rating != 'NaN'
)
SELECT
    min(US_URLs.share_title) as share_title,
    min(US_URLs.share_main_blurb) as share_main_blurb,
    urlbd.url_rid AS url_rid,
    min(US_URLs.clean_url) as clean_url,
    min(US_URLs.parent_domain) as parent_domain,
    min(US_URLs.tpfc_rating) as tpfc_rating,
    min(US_URLs.tpfc_first_fact_check) as tpfc_first_fact_check,
    min(urlbd.year_month) as year_month,
    min(urlbd.political_page_affinity) as political_page_affinity,
    SUM(urlbd.share_without_clicks) AS share_without_clicks,
    SUM(urlbd.shares) AS shares,
    SUM(urlbd.clicks) AS clicks
FROM {database}.{breakdowns_table} urlbd
JOIN US_URLs ON
urlbd.url_rid = US_URLs.url_rid
WHERE urlbd.c='US' and urlbd.political_page_affinity = '2'
GROUP BY urlbd.url_rid
"""

dfpos2 = execute(sql)
```

## 12. Combine all data with 5 user affinity scores and merge it with politically classified data
```
frames = [dfneg2, dfneg1, dfzero, dfpos1, dfpos2]
df_affinity = pd.concat(frames)
df_pol = pd.read_csv(r'tpfc_pol.csv', delimiter = ',')
result = pd.merge(df_affinity, df_pol, on='url_rid')
```

## 13. Filter and analyze political-only URLs (for non-political content set political == 0)
```
df = result[result['political'] == 1]
df = pd.DataFrame({'count' : df.groupby('political_page_affinity').size(), 
                   'share_without_clicks': df.groupby('political_page_affinity')['share_without_clicks'].sum(),
                   'shares': df.groupby('political_page_affinity')['shares'].sum(),
                   'clicks': df.groupby('political_page_affinity')['clicks'].sum(),
                  }).reset_index()
```

## 14. Filter and analyze political-only URLs with fake news (for true news set tpfc_rating == ‘fact chekced as true’)
```
from numpy import sqrt
from numpy import power

df = result[result['political'] == 1]
df = df[df['tpfc_rating'] == 'fact checked as false']
df = pd.DataFrame({'count' : df.groupby('political_page_affinity').size(), 
                   'share_without_clicks': df.groupby('political_page_affinity')['share_without_clicks'].sum(),
                   'shares': df.groupby('political_page_affinity')['shares'].sum(),
                   'clicks': df.groupby('political_page_affinity')['clicks'].sum(),
                  }).reset_index()

df['SwoC_LLCI']=round(df['share_without_clicks']-1.96*sqrt(df['count']*power(10,2)))
df['SwoC_ULCI']=round(df['share_without_clicks']+1.96*sqrt(df['count']*power(10,2)))
df['Shares_LLCI']=round(df['shares']-1.96*sqrt(df['count']*power(14,2)))
df['Shares_ULCI']=round(df['shares']+1.96*sqrt(df['count']*power(14,2)))
df['Clicks_LLCI']=round(df['clicks']-1.96*sqrt(df['count']*power(40,2)))
df['Clicks_ULCI']=round(df['clicks']+1.96*sqrt(df['count']*power(40,2)))
```
