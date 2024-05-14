import sys
sys.path.insert(0, '../nsga-net/')
from search import train_search
import os
from datetime import datetime
genome_string = [[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
start_time = datetime.now()
performance = train_search.main(genome=genome_string,
                                            search_space = 'macro',
                                            init_channels = 36,
                                            layers=15, cutout=True,
                                            epochs=20,
                                            save='arch_{}'.format(1),
                                            expr_root='')
# Get the current time after running the code
end_time = datetime.now()

# Calculate the duration it took to run the code
duration = end_time - start_time
print(f"The code took {duration} to run.")
