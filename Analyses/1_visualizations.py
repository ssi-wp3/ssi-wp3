import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import config

if __name__ == "__main__":
  dataset_fn = "lidl_ah_jumbo_plus.parquet"
  dataset_path = os.path.join(config.OUTPUT_DATA_DIR, dataset_fn)

  df_stores = pd.read_parquet(dataset_path)

  # set up output directory for graphics
  if not os.path.isdir(config.OUTPUT_GRAPHICS_DIR):
    os.mkdir(config.OUTPUT_GRAPHICS_DIR)

  #
  # number of records per month for each supermarket
  # x-axis: months, y-axis stores
  #
  x_axis_col = "year_month"
  y_axis_col = "store_name"
  val_col    = "count"

  graph_data = df_stores.groupby([x_axis_col, y_axis_col])[val_col].sum()
  graph_data = graph_data.reset_index()
  graph_data = graph_data.pivot(index=x_axis_col, columns=y_axis_col)
  graph_data = graph_data.fillna(0)
  graph_data = graph_data.transpose() # flip x and y such that year_month is displayed at the bottom


  sns.heatmap(graph_data, cmap="Blues", robust=True)

  out_fn = "stores__inventory_per_month.png"
  plt.savefig(os.path.join(config.OUTPUT_GRAPHICS_DIR, out_fn))

  # graphic for cells greater than 0
  graph_data_bin = graph_data > 0

  plt.clf()
  sns.heatmap(graph_data_bin, cmap="Blues")

  out_fn = "stores__inventory_per_month_bin.png"
  plt.savefig(os.path.join(config.OUTPUT_GRAPHICS_DIR, out_fn))



  

  
