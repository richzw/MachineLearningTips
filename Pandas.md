- [Pandas Groupby](https://mp.weixin.qq.com/s/hFTAXCP2DPsras1Ewo5TVA)
  - 单列聚合 `sales.groupby("store")["stock_qty"].mean()`
  - 多列聚合 `sales.groupby("store")[["stock_qty","price"]].mean()`
  - 多列多个聚合 `sales.groupby("store")["stock_qty"].agg(["mean", "max"])`
  - 对聚合结果进行命名  
    ```
    sales.groupby("store").agg(  
      avg_stock_qty = ("stock_qty", "mean"),
      max_stock_qty = ("stock_qty", "max")
     )
     ```
  - 多个聚合和多个函数  `sales.groupby("store")[["stock_qty","price"]].agg(["mean", "max"])`
  - as_index参数 
    - 如果groupby操作的输出是DataFrame，可以使用as_index参数使它们成为DataFrame中的一列。
      ```
      sales.groupby("store", as_index=False).agg(
     
      avg_stock_qty = ("stock_qty", "mean"),
      avg_price = ("price", "mean")
      )
      ```
 




