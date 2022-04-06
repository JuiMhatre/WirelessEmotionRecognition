import pandas as pd

# List of Tuples
dataset = pd.read_excel('C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/readfile.xlsx',
                                index_col=None,
                                usecols=['mean_rr', 'median_rr', 'sdrr', 'rmssd', 'sdsd', 'sdrr_rmssd', 'hr', 'pnn25',
                                         'pnn50', 'sd1', 'sd2', 'emotion'])



# Selecting duplicate rows based
# on 'City' column

duplicate_emotion=dataset.groupby(['mean_rr', 'median_rr', 'sdrr', 'rmssd', 'sdsd', 'sdrr_rmssd', 'hr', 'pnn25',
                                         'pnn50', 'sd1', 'sd2', 'emotion'],as_index=False).size()
duplicate=dataset.groupby(['mean_rr', 'median_rr', 'sdrr', 'rmssd', 'sdsd', 'sdrr_rmssd', 'hr', 'pnn25',
                                         'pnn50', 'sd1', 'sd2'],as_index=False).size()

df1 = (dataset.groupby(['mean_rr', 'median_rr', 'sdrr', 'rmssd', 'sdsd', 'sdrr_rmssd', 'hr', 'pnn25',
                                         'pnn50', 'sd1', 'sd2'])        .apply(lambda x: tuple(x.index))        .reset_index(name='idx'))
emot = dataset['emotion'][[tuple(sorted(x))[0] for x in df1['idx']]]
duplicate['emotion'] = list(emot)
# list1 =[]
# for tup in df1['idx']:
#     list1.append(list(tup))
# list1 =[item for sublist in list1 for item in sublist]
duplicate.to_csv('C:/Users/16786/PycharmProjects/EmotionRecognition/dataset/clean.csv', index=False)
print("done")


# Print the resultant Dataframe
