

def _levenshtein_distance_matrix(string1, string2, is_damerau=False):
    n1 = len(string1)
    n2 = len(string2)
    d = np.zeros((n1 + 1, n2 + 1), dtype=int)
    for i in range(n1 + 1):
        d[i, 0] = i
    for j in range(n2 + 1):
        d[0, j] = j
    for i in range(n1):
        for j in range(n2):
            if string1[i] == string2[j]:
                cost = 0
            else:
                cost = 1
            d[i+1, j+1] = min(d[i, j+1] + 1, # insert
                              d[i+1, j] + 1, # delete
                              d[i, j] + cost) # replace
            if is_damerau:
                if i > 0 and j > 0 and string1[i] == string2[j-1] and string1[i-1] == string2[j]:
                    d[i+1, j+1] = min(d[i+1, j+1], d[i-1, j-1] + cost) # transpose
    return d

def levenshtein_distance(string1, string2):
    n1 = len(string1)
    n2 = len(string2)
    return _levenshtein_distance_matrix(string1, string2)[n1, n2]


def damerau_levenshtein_distance(string1, string2):
    n1 = len(string1)
    n2 = len(string2)
    return _levenshtein_distance_matrix(string1, string2, True)[n1, n2]


def get_ops(string1, string2, is_damerau=False):
    dist_matrix = _levenshtein_distance_matrix(string1, string2, is_damerau)
    i, j = dist_matrix.shape
    i -= 1
    j -= 1
    ops = list()
    while i != -1 and j != -1:
        if is_damerau:
            if i > 1 and j > 1 and string1[i-1] == string2[j-2] and string1[i-2] == string2[j-1]:
                if dist_matrix[i-2, j-2] < dist_matrix[i, j]:
                    ops.insert(0, ('transpose', i - 1, i - 2))
                    i -= 2
                    j -= 2
                    continue
        index = np.argmin([dist_matrix[i-1, j-1], dist_matrix[i, j-1], dist_matrix[i-1, j]])
        if index == 0:
            if dist_matrix[i, j] > dist_matrix[i-1, j-1]:
                ops.insert(0, ('replace', i - 1, j - 1))
            i -= 1
            j -= 1
        elif index == 1:
            ops.insert(0, ('insert', i - 1, j - 1))
            j -= 1
        elif index == 2:
            ops.insert(0, ('delete', i - 1, i - 1))
            i -= 1
    return ops

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

def predict_typo(domain_name,n):
    global token_size
    typo_dict = dict()
    for i in range(len(domain_name)-token_size+1):
        substr = domain_name[i:i+token_size]
        substr = '\t'+substr+'\n'
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


def tokenize(reg_list, typo_list, token_size):
    in_list = list()
    out_list = list()
    for i in range(len(reg_list)):
        ops = get_ops(reg_list[i],typo_list[i])
        if len(ops) == 0:
            continue
        if ops[0][0] == 'replace':
            for start_idx in range(ops[0][1]-token_size+1,ops[0][1]+1):
                if start_idx < 0 :
                    continue
                if start_idx > (len(reg_list[i])-token_size):
                    break
                in_list.append('\t'+reg_list[i][start_idx:start_idx+token_size]+'\n')
                out_list.append('\t'+typo_list[i][start_idx:start_idx+token_size]+'\n')

        elif ops[0][0] == 'insert':
            for start_idx in range(ops[0][1]-token_size+1,ops[0][1]+1):
                if start_idx < 0:
                    continue
                if start_idx > (len(reg_list[i])-token_size):
                    continue
                in_list.append('\t'+reg_list[i][start_idx:start_idx+token_size]+'\n')
                out_list.append('\t'+typo_list[i][start_idx:start_idx+token_size+1])
            for start_idx in range(ops[0][1]+1,ops[0][1]+2):
                if start_idx > (len(reg_list[i])-token_size):
                    continue
                in_list.append('\t'+reg_list[i][start_idx:start_idx+token_size]+'\n')
                out_list.append(typo_list[i][start_idx:start_idx+token_size+1]+'\n')


        elif ops[0][0] == 'delete':
            tt = reg_list[i][:ops[0][1]]+'*'+typo_list[i][ops[0][1]:]
            for start_idx in range(ops[0][1]-token_size+1,ops[0][1]+1):
                if start_idx < 0 :
                    continue
                if start_idx > (len(reg_list[i])-token_size):
                    break
                in_list.append('\t'+reg_list[i][start_idx:start_idx+token_size]+'\n')
                out_list.append('\t'+tt[start_idx:start_idx+token_size]+'\n')
        return in_list, out_list
    
class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        s=''
        for i in range(x.shape[-1]):
            s=s+self.indices_char[x[0,i]]
        return s
