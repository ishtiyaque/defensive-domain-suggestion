import utils
import pandas as pd
import numpy as np

from keras.models import load_model

def predict_typo(domain_name,n):
    global token_size
    typo_dict = dict()
    for i in range(len(domain_name)-token_size+1):
        substr = domain_name[i:i+token_size]
        substr = '\t'+substr+'\n'
		model = load_model('typo_model.h5')
        pred=model.predict(table.encode(substr,token_size+2).reshape((1,token_size+2,len(vocab))))
        d=get_pred(pred,n)
        ss = sorted(d,reverse=True)
        for val in ss[:n]:
            pred_sub_str = d[val]
            if pred_sub_str[-1] == '\n':
                pred_sub_str = pred_sub_str[:-1]
            if pred_sub_str[0] == '\t':
                pred_sub_str = pred_sub_str[1:]
            pred_typo = domain_name[:i]+pred_sub_str+domain_name[i+token_size:]
            typo_dict[val]=pred_typo
    return typo_dict

def get_pred(pred,depth=2):
    depth=15
    sorted_idx = np.argsort(-pred)
    d = dict()
    for i0 in range(depth):
        idx0 = sorted_idx[0,0,i0]
        p0 = pred[0,0,idx0]
        c0 = table.indices_char[idx0]

        for i1 in range(depth):
            idx1 = sorted_idx[0,1,i1]
            p1 = p0 * pred[0,1,idx1]
            c1 = c0+table.indices_char[idx1]
            #print c1

            for i2 in range(depth):
                idx2 = sorted_idx[0,2,i2]
                p2 = p1 * pred[0,2,idx2]
                c2 = c1+table.indices_char[idx2]
                #print c2,p2

                for i3 in range(depth):
                    idx3 = sorted_idx[0,3,i3]
                    p3 = p2 * pred[0,3,idx3]
                    c3 = c2+table.indices_char[idx3]
                    #print c3,p3

                    for i4 in range(depth):
                        idx4 = sorted_idx[0,4,i4]
                        p4 = p3 * pred[0,4,idx4]
                        c4 = c3+table.indices_char[idx4]
                        d[p4]=c4
    return d  


token_size = 3
url="mynewurl"
typo_dict = predict_typo(url,5)
sorted_typo_dict = sorted(typo_dict,reverse=True)

unique_set = set()
typo_dict = predict_typo(url, 5)
sorted_typo_dict = sorted(typo_dict,reverse=True)
for prob in sorted_typo_dict[:5]:
	if typo_dict[prob] != url:
		unique_set.add(typo_dict[prob])
	if len(unique_set) == num_unique:
		break

print unique_set