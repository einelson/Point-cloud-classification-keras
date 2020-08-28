'''
Notes
PPtk viewer
https://heremaps.github.io/pptk/viewer.html

laspy
https://pythonhosted.org/laspy/
'''



'''
File data layout
points = pd.DataFrame(columns=['x', 'y', 'z', 'class'])
'''


import pandas as pd
import numpy as np
from laspy.file import File



# #open the file
# lasFile = File("./data/Room1_filtered.las", mode = "rw")

# #put points in numpy array
# lasPoints = lasFile.points

# print(lasPoints)

# # df = pd.DataFrame(lasPoints)

# # print(df)
las_header = None
max_points=1000000000



# with File('./data/Room1_filtered.las') as f:
#     if las_header is None:
#         las_header = f.header.copy()
#     if max_points is not None and max_points < f.header.point_records_count:
#         # mask = Mask(f.header.point_records_count, False)
#         mask[np.random.choice(f.header.point_records_count, max_points)] = True
#     else:
#         # mask = Mask(f.header.point_records_count, True)
#         new_df = pd.DataFrame(np.array((f.x, f.y, f.z)).T[mask.bools])
#         new_df.columns = ['x', 'y', 'z']
#     if f.header.data_format_id in [2, 3, 5, 7, 8]:
#         rgb = pd.DataFrame(np.array((f.red, f.green, f.blue), dtype='int').T[mask.bools])
#         rgb.columns = ['r', 'g', 'b']
#         new_df = new_df.join(rgb)
#     new_df['class'] = f.classification[mask.bools]
#     if np.sum(f.user_data):
#         new_df['user_data'] = f.user_data[mask.bools].copy()
#     if np.sum(f.intensity):
#         new_df['intensity'] = f.intensity[mask.bools].copy()