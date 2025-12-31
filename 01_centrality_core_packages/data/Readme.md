—- DAG Dataset Summary —
698903 total number of nodes in nodes.csv
1819937 total number of edges in edges.csv
397798 total number of nodes which is an endpoint of at least one edge (nodes in edges.csv)

1. Loading edges from edges.csv...
   > Using columns: Source='source', Target='target'
2. Constructing Directed Graph (DiGraph)...
   > Graph Stats: 397,798 nodes, 1,819,937 edges.

   Here is the DAG dataset that you can use in your tasks friends, the edges.csv is formatted in ‘source, target’ format where the first column is the source node and the second column is the target node of the directed edge. These two files (nodes.csv and edges.csv) should be sufficient for all applications and they already have the header lines in CSV files such as ‘source,target’ in edges.csv, so it should be clear.