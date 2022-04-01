def get_weights(df):
  label_weights = [0]*2
  lm = df['labels'].value_counts().reset_index().values.tolist()
    
  for idx, (e, c) in enumerate(lm):
    label_weights[e] = c

  return  label_weights