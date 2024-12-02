def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor

def cond_probs_product(full_table, evidence_row, target_column, target_column_value):
  assert target_column in full_table
  assert target_column_value in up_get_column(full_table, target_column)
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1   # - 1 because subtracting off the target column from full_table

  #your function body below
  #print(evidence_row)

  table_columns = up_list_column_names(full_table)
  evidence_columns = table_columns[:-1]
  evidence_complete = up_zip_lists(evidence_columns, evidence_row)  #new puddles function

  cond_prob_list = [cond_prob(full_table, x[0], x[1], target_column, target_column_value) for x in evidence_complete]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(full_table, the_column, the_column_value):
  assert the_column in full_table
  assert the_column_value in up_get_column(full_table, the_column)

  #your function body below
  t_list = up_get_column(full_table, the_column)
  p_a = sum([1 if v==the_column_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(full_table, evidence_row, target_column):
  assert target_column in full_table
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1   # - 1 because subtracting off the target

  #compute P(target=0|...) by using cond_probs_product, finally multiply by P(target=0) using prior_prob
  P_0 = cond_probs_product(full_table, evidence_row, target_column, 0) * prior_prob(full_table, target_column, 0)

  #do same for P(target=1|...)
  P_1 = cond_probs_product(full_table, evidence_row, target_column, 1) * prior_prob(full_table, target_column, 1)

  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(P_0, P_1)
  #return your 2 results in a list
  return [neg, pos]

def metrics(zipped_list):
  assert isinstance(zipped_list, list)
  assert all([isinstance(v, list) for v in zipped_list])
  assert all([len(v)==2 for v in zipped_list])
  assert all([isinstance(a,(int,float)) and isinstance(b,(int,float)) for a,b in zipped_list]), f'zipped_list contains a non-int or non-float'
  assert all([float(a) in [0.0,1.0] and float(b) in [0.0,1.0] for a,b in zipped_list]), f'zipped_list contains a non-binary value'

  #first compute the sum of all 4 cases. See code above
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  accuracy = (tp + tn) / len(zipped_list) if len(zipped_list) > 0 else 0
  precision = tp / (tp + fp) if (tp + fp) > 0 else 0
  recall = tp / (tp + fn) if (tp +fn) > 0 else 0
  f1 = 2*((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

  #now build dictionary with the 4 measures - round values to 2 places
  my_dict = {'Precision': round(precision, 2), 'Recall': round(recall, 2), 'F1': round(f1, 2), 'Accuracy': round(accuracy, 2)}

  #finally, return the dictionary
  return my_dict

from sklearn.ensemble import RandomForestClassifier
def run_random_forest(train, test, target, n):
  #target is target column name
  #n is number of trees to use
 
  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)

  assert target in train   #have not dropped it yet
  assert target in test

  X = up_drop_column(train, target)
  y = up_get_column(train, target)
  assert isinstance(y,list)
  assert len(y)==len(X)
  clf.fit(X, y)

  k_feature_table = up_drop_column(test, target)
  k_actuals = up_get_column(test, target)

  probs = clf.predict_proba(k_feature_table)  #Note no need here to transform k_feature_table to list - we can just use the table. Nice.

  assert len(probs)==len(k_actuals)
  assert len(probs[0])==2

  pos_probs = [p for n,p in probs]  #just the positive probabilities 

  thresholds = [.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95] 

  all_mets = []
  for t in thresholds:
    predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  metrics_table = up_metrics_table(all_mets)

  return metrics_table

def try_archs(train_table, test_table, target_column_name, architectures, thresholds):
  arch_acc_dict = {}  #ignore if not attempting extra credit

  for arch in architectures:

    probs = up_neural_net(train_table, test_table, arch, target_column_name)
    pos_probs = [x[1] for x in probs]
    all_mets = []
    for t in thresholds:
      predictions = [1 if pos>=t else 0 for pos in pos_probs]
      pred_act_list = up_zip_lists(predictions, up_get_column(test_table, target_column_name))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    metrics_table = up_metrics_table(all_mets)

    #arch_acc_dict[tuple(arch)] = max(...)  #extra credit - uncomment if want to attempt

    print(f'Architecture: {arch}')
    display(up_metrics_table(all_mets))

  return arch_acc_dict
