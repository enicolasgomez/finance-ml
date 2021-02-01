import seaborn as sn
import matplotlib.pyplot as plt 

def get_correlative_pairs(dataset):
  corr_matrix = dataset.corr().abs()
  sn.heatmap(corr_matrix, annot=True)
  plt.show()
  corr_pairs = dataset.corr().unstack().sort_values().drop_duplicates()
  corr_list = [i[0] for i in corr_pairs.items()]
  return corr_list