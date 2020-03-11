import xlrd
import pandas as pd
import xlsxwriter
import os
import math

index = 1


file_name = "sampleset.xlsx"
workbook = xlsxwriter.Workbook(file_name)
worksheet = workbook.add_worksheet('sampleset')
worksheet.write_row(0, 0, ['text', 'entity', 'verb','e1','e2','relation1','e1','e2','relation2','e1','e2','relation3'])

path = "/Users/charles/Desktop/CZYFYP/Annotated/"

for read_filename in os.listdir(path):
    print(read_filename)
    if read_filename[-5:] != '.xlsx':
        continue
    iiii = 0
    df = pd.read_excel(path+read_filename)
    data = df.values
    for line in data:
        #print(iiii)
        iiii = iiii + 1
        end = -1
        for i in line:
            end = end+1
            if pd.isnull(i):
                break

        worksheet.write_row(index, 0,line.tolist()[:end])
        #worksheet.write_row(1, 0, line.tolist())
        index = index + 1




workbook.close()
