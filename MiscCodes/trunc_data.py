# truncate data files bya time

file_in = 'E:/BAT/061118_flow_cyto_v1/neg_stim_Av_0pt1_0pt3_20000sps.txt'
file_out = 'E:/BAT/061118_flow_cyto_v1/neg_stim_Av_0pt1_0pt3_20000sps_trunc_v2.txt'


with open(file_in,'r') as infile, open(file_out,'w') as outfile:
    for i,line in enumerate(infile):
        time = float(line.split()[0])
        if time < 3700:
            outfile.write(line)
            if i % 5000000 == 0: print('writing line ',i)
        else:
            if i % 10000000 == 0: print('omitting line ',i)
            
