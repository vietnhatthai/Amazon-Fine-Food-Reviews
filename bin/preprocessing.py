import os
import re
import pandas as pd

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    '''
    remove html tags from text
    '''
    return TAG_RE.sub('', text)

root = 'data'

data = pd.read_csv(os.path.join(root, 'Reviews.csv'))

data['Rating'] = data['Score'].astype(int)

data.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator',
       'HelpfulnessDenominator', 'Score', 'Time', 'Summary'], axis=1, inplace=True)

data['Score'] = data['Rating'].apply(lambda x : x-1).astype(float) / 4  # normalize to [0, 1]
data['Text'] = data['Text'].apply(remove_tags)  

# train 70 %, test 30 %
train = data.sample(frac=0.7, random_state=42).reset_index(drop=True)
test = data.drop(train.index).reset_index(drop=True)

# save train and test data
train.to_csv(os.path.join(root, 'train.csv'), index=False)
test.to_csv(os.path.join(root, 'test.csv'), index=False)