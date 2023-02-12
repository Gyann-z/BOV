# -*- coding:utf-8 -*-

import time
import pypyodbc as mdb

connStr = (r'Driver={Microsoft Access Driver (*.mdb)};DBQ=/data2/zgy_data/TextVQA/mmf/data/datasets/textvqa/defaults/features/open_images/detectron.lmdb/lock.mdb;'
           r'Database=bill;'
           )
conn = mdb.win_connect_mdb(connStr)
cur = conn.cursor()

cur.execute('SELECT * FROM bill;')

for col in cur.description:
    # 展示行描述
    print(col[0], col[1])
result = cur.fetchall()

for row in result:
    # 展示个字段的值
    print(row)
    print(row[1], row[2])


