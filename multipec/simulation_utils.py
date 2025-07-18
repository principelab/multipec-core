import numpy as np
from random import choice, shuffle
from itertools import permutations, combinations
from copy import deepcopy
from string import ascii_letters, punctuation

from multipec.data_legacy import create, binarize
from multipec.awc import AWC

import matplotlib.pyplot as plt
import seaborn as sns

def set_plotting_style(context="paper", figsize=(8, 6), dpi=100):
    """
    Set global matplotlib and seaborn plotting style.

    Parameters:
    - context: seaborn context ("paper", "notebook", "talk", "poster")
    - figsize: tuple, default figure size in inches
    - dpi: int, resolution for display (use 300 when saving)
    """
    sns.set(style="whitegrid", context=context)

    plt.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,

        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 15,
        "legend.title_fontsize": 16,

        "lines.linewidth": 3,
        "lines.markersize": 8,

        "legend.frameon": False,
        "legend.loc": "best",

        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",

        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def flatten(object):
    """Flattens nested iterables.

    Args:
        object (any iterable type): general list, tuple or set

    Yields:
        any type: nested elements of the object
    """
    for item in object:
        if isinstance(item, (list, tuple, set)):
            yield from flatten(item)
        else:
            yield item

def random_stimulus(l, pool, pool1=False):
    """Creates a string sequence of l random characters from a pool.
    Possible to create a sequence of random characters mixed from pool and pool1, with the same ratio.

    Args:
        l (int): sequence length
        pool (list): list of characters
        pool1 (bool, list): second list of characters. Defaults to False.

    Returns:
        string: random seuqence of characters
    """
    s=""
    if not pool1:
        for i in range(l): s+=choice(pool)
    else:
        order=[0]*int(l/2)+[1]*int(l/2)
        shuffle(order)
        for i in range(l):
            if not order[i]: s+=choice(pool)
            elif order[i]: s+=choice(pool1)
    return s

def recode_to_6(char):
    tab={"upper_case":["[","]","^","_","@","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
    "lower_case":['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "`","{","|","}","~"],
    "n_0":list(range(1,16)),"n_1":list(range(16,32)),"n_2":list(range(32,64)),"n_2p":[":",";","<","=",">","?","-"],"n_3":list(range(64,127))}
    if char in tab["upper_case"]: return str(list(tab["n_0"]+tab["n_1"])[tab["upper_case"].index(char)])
    elif char.isnumeric() and 0<=int(char)<=9: return str(tab["n_2"][int(char)])
    elif char in tab["n_2p"]: return char
    else: return "Error: character out of range"

def to_binary(obj):
    if str(obj).isnumeric() or type(obj) in (int, float, complex): 
        nobj=int(obj)
        return list(int(b) for b in '{0:08b}'.format(nobj))
    else: return [int(b) for b in list(''.join(format(ord(obj), '08b')))]

def sim3nets(seq, area1, area2, area3, fs, seq_multiplier, base="random", random_signal=ascii_letters, bits=6):
    seq1,bb,x=[],[],8-bits
    if base=="random": bb=list(choice(punctuation))
    else: bb=list(base)
    
    for r in range(seq_multiplier):
        shuffle(seq)
        for s in seq:
            seq1+=[s]*fs
            seq1+=base
    if bits==5:
        a1,a2,a3,a4,a5 = [[] for _ in range(bits)]
        b1,b2,b3,b4,b5 = [[] for _ in range(bits)]
        c1,c2,c3,c4,c5 = [[] for _ in range(bits)]
        a_channels,b_channels,c_channels = [a1,a2,a3,a4,a5],[b1,b2,b3,b4,b5],[c1,c2,c3,c4,c5] 
    if bits==6:
        a1,a2,a3,a4,a5,a6 = [[] for _ in range(bits)]
        b1,b2,b3,b4,b5,b6 = [[] for _ in range(bits)]
        c1,c2,c3,c4,c5,c6 = [[] for _ in range(bits)]
        a_channels,b_channels,c_channels = [a1,a2,a3,a4,a5,a6],[b1,b2,b3,b4,b5,b6],[c1,c2,c3,c4,c5,c6] 

    for el in seq1:
        if el in bb:
            for unit in el:
                recoded=recode_to_6(unit)
                baseline=to_binary(recoded)[x::]
                for j,b in enumerate(baseline): 
                    a_channels[j].append(b)
                    b_channels[j].append(b)
                    c_channels[j].append(b)
        elif el in area1 and el not in area2+area3:
            for unit in el:
                recoded=recode_to_6(unit)
                byte=to_binary(recoded)[x::]
                rand_byte1=to_binary(choice(random_signal))[x::]
                rand_byte2=to_binary(choice(random_signal))[x::]
                for j,b in enumerate(byte): 
                    a_channels[j].append(b)
                    b_channels[j].append(rand_byte1[j])
                    c_channels[j].append(rand_byte2[j])
        elif el in area1 and el in area3 and el not in area2:
            for unit in el:
                recoded=recode_to_6(unit)
                byte=to_binary(recoded)[x::]
                rand_byte1=to_binary(choice(random_signal))[x::]
                for j,b in enumerate(byte): 
                    a_channels[j].append(b)
                    b_channels[j].append(rand_byte2[j])
                    c_channels[j].append(b)
        elif el in area2 and el not in area1+area3:
            n,recoded_el=0,""
            for unit in el: recoded_el+=recode_to_6(unit)
            while n<len(recoded_el):
                byte=to_binary(int(recoded_el[n:n+2]))[x::]
                rand_byte2=to_binary(choice(random_signal))[x::]
                rand_byte1=to_binary(choice(random_signal))[x::]
                for j,b in enumerate(byte):
                    a_channels[j].append(rand_byte1[j])
                    b_channels[j].append(b)
                    c_channels[j].append(rand_byte2[j])  
                n+=2
        elif el in area3 and el not in area1+area2:
            n=0
            while n<len(el):
                recoded=recode_to_6(el[n])
                byte=to_binary(recoded)[x::]
                rand_byte1=to_binary(choice(ascii_letters))[x::]
                rand_byte2=to_binary(choice(ascii_letters))[x::]
                for j,b in enumerate(byte):
                    a_channels[j].append(rand_byte1[j])
                    b_channels[j].append(rand_byte2[j])
                    c_channels[j].append(b)   
                n+=1
        elif el in area2 and el in area3 and el not in area1:
            n,recoded_el=0,""
            for unit in el: recoded_el+=recode_to_6(unit)
            while n<len(recoded_el):
                byte=to_binary(int(recoded_el[n:n+2]))[x::]
                rand_byte1=to_binary(choice(random_signal))[x::]
                for j,b in enumerate(byte):
                    a_channels[j].append(rand_byte1[j])
                    b_channels[j].append(b)
                    c_channels[j].append(b)  
                n+=2
    return seq1,a_channels,b_channels,c_channels

def sim2nets(seq, area1, area2, fs, seq_multiplier, base="random", random_signal=ascii_letters, bits=6):
    seq1,bb,x=[],[],8-bits
    if base=="random": bb=list(choice(punctuation))
    else: bb=list(base)
    
    for r in range(seq_multiplier):
        shuffle(seq)
        for s in seq:
            seq1+=[s]*fs
            seq1+=base
    
    if bits==5:
        a1,a2,a3,a4,a5 = [[] for _ in range(bits)]
        b1,b2,b3,b4,b5 = [[] for _ in range(bits)]
        a_channels,b_channels = [a1,a2,a3,a4,a5],[b1,b2,b3,b4,b5]
    if bits==6:
        a1,a2,a3,a4,a5,a6 = [[] for _ in range(bits)]
        b1,b2,b3,b4,b5,b6 = [[] for _ in range(bits)]
        a_channels,b_channels = [a1,a2,a3,a4,a5,a6],[b1,b2,b3,b4,b5,b6]

    for el in seq1:
        if el in bb:
            for unit in el:
                recoded=recode_to_6(unit)
                baseline=to_binary(recoded)[x::]
                for j,b in enumerate(baseline): 
                    a_channels[j].append(b)
                    b_channels[j].append(b)
        elif el in area1 and el not in area2:
            for unit in el:
                recoded=recode_to_6(unit)
                byte=to_binary(recoded)[x::]
                rand_byte1=to_binary(choice(random_signal))[x::]
                for j,b in enumerate(byte): 
                    a_channels[j].append(b)
                    b_channels[j].append(rand_byte1[j])
        elif el in area2 and el not in area1:
            n,recoded_el=0,""
            for unit in el: recoded_el+=recode_to_6(unit)
            while n<len(recoded_el):
                byte=to_binary(int(recoded_el[n:n+2]))[x::]
                rand_byte1=to_binary(choice(random_signal))[x::]
                for j,b in enumerate(byte):
                    a_channels[j].append(rand_byte1[j])
                    b_channels[j].append(b)
                n+=2
    return seq1,a_channels,b_channels

def moving_average(x, w):
    if len(x)>1: return np.convolve(x, np.ones(w), 'valid') / w
    else: return x

def threshold(errors_list, sigmas):
    if len(errors_list)==1: return errors_list[0]
    else:
        av = np.average(errors_list)
        std = np.std(errors_list)
        return float(av+std*sigmas)

def PEC_pairs(channels_dict):
    subset = list(channels_dict.keys())
    ch_ids,register = list(permutations(list(subset), 2)),{}
    for c in ch_ids:
        group = list(channels_dict[ci] for ci in c)
        data = create(group)
        awc = AWC(bits=2, time=6)
        R = awc.learn(data.serialized)
        r = np.average(R['error'].deserialized[1])
        register[c]=float(r)
        # print("processed... ",c," error: ",register[c])
    return(register)

def direct(register, channels):
    subset = channels
    directed_pairs = {}
    for p in list(combinations(subset,2)):
        if register[(p[0],p[1])] < register[(p[1],p[0])]: directed_pairs[p]=register[(p)]
        elif register[(p[0],p[1])] > register[(p[1],p[0])]: directed_pairs[(p[1],p[0])]=register[(p[1],p[0])]
        else: directed_pairs[p]=register[(p)]; directed_pairs[(p[1],p[0])]=register[(p[1],p[0])]; print(p)
    return directed_pairs

def PEC_multi(subset, n, seeds, nodes):
    ch_ids,register,candidates_ids = permutations(list(subset), n),{},[]
    for p in ch_ids:
        if p[:-1]==nodes: candidates_ids.append(p)
    for c in candidates_ids:
        group = list(seeds[ci] for ci in c)
        data = create(group)
        awc = AWC(bits=n, time=6)
        R = awc.learn(data.serialized)
        r = np.average(R['error'].deserialized[1])
        register[c]=float(r)
    return register

def nets_from_pairs(directed_pairs, labels_list, sigmas):
    ''' recursively builds networks based on the minimal error from pairs;
    previously selected PAIRS are not considered in the next iteration;
    WORKS BEST '''
    unassigned = deepcopy(directed_pairs)
    count0 = len(directed_pairs)
    subnets,output,output_idx = [],{},{}
    errors_log,output_err = [],{}
    while count0>0:
        nodes,result,count1=[],[],len(unassigned)
        while count1>0:
            next = deepcopy(unassigned)
            next_keys = list(next.keys())
            if nodes:
                for i in next_keys:
                    # Only keep pairs that share a node with the last added one
                    if i[0] not in nodes[-1] and i[1] not in nodes[-1]: del next[i]
                for node in nodes: 
                    if node in next: del next[node]
            if len(next)>1:
                # Add the pair with the lowest error
                result.append(min(next.items(), key=lambda x: x[1]))
                errors = list(tup[1] for tup in result[:-1])
                if len(result)>2 and result[-1][1]>threshold(errors,sigmas):
                    # If the most recent added pair has error higher than a computed threshold, it's removed and the subnet ends
                    print(result[-1])
                    result.pop(-1)
                    subnets.append(nodes)
                    errors_log.append(errors)
                    count1=0; break
                else: 
                    nodes.append(result[-1][0])
                    count1-=1
            elif len(next)==1:
                    print("1 node unassigned: ")
                    print(next) 
                    subnets.append(list(next.keys()))
                    errors_log.append(list(next.values()))
                    count0,count1=0,0; break
            elif not next: print("no nodes left"); count1=0
        selected_pairs=list(r[0] for r in result)
        selected_nodes=list(set(flatten(subnets)))
        if len(selected_nodes)==len(labels_list): count0=0
        else:
            keys=list(unassigned.keys())
            if not keys: count0=0
            elif keys:
                for t in keys:
                    for pair in selected_pairs:
                        if t[0] in pair and t[1] in pair:
                            # Ensures pairs already used aren't reused in new subnets
                            count0-=1
                            del unassigned[t]
    for j,sub in enumerate(subnets): 
        output[j]=[(labels_list[x[0]],(labels_list[x[1]])) for x in sub]
        output_idx[j]=tuple(set([labels_list[nd] for pair in sub for nd in pair]))
        output_err[j]=errors_log[j]
    return output_idx, output, output_err

def subnets_1i(channels_dict, labels_list, directed_pairs, nonselected_nodes, limit=False):
    '''iterative version;
    selects functional subnetworks from a pool of nodes;
    each node can belong to one subnetwork;
    process ends when all nodes are selected'''
    seeds, pairs, pool = deepcopy(channels_dict), deepcopy(directed_pairs), deepcopy(nonselected_nodes)
    output = []
    while pool:
        if len(pool)==1:
            subnet = pool
            output.append(subnet)
            print("1 unassigned node: ", pool[0])
            pool.pop()
            return output
        else:
            print("starting a new subnetwork... ")
            n, connectivity, nodes, result = 2, {}, [], []
            subset = deepcopy(pool) #continue with the nonselected nodes  

            n_max = len(subset) 
            if type(limit)==int: 
                n_max = limit if limit<len(subset) else len(subset)  
                print(f'network size limit = {n_max}')  
                
            while n<=n_max:
                if n==2: connectivity = {i:pairs[i] for i in pairs if i[0] in subset and i[1] in subset}
                elif n>2: connectivity = PEC_multi(subset, n, seeds, nodes)
                result.append(min(connectivity.items(), key=lambda x: x[1]))
                errors = list(tup[1] for tup in result)
                mav_errors = moving_average(errors, 2)
                if len(mav_errors)>1 and mav_errors[-1]>mav_errors[-2]: 
                    print("moving average increased ", mav_errors[-2], "<", mav_errors[-1])
                    del result[-1]
                    nodes=result[-1][0]
                    n=n_max+1
                else: 
                    nodes=result[-1][0]
                    n+=1
                    print("adding nodes...")

            subnet = [labels_list[nd] for nd in nodes]
            print(result, subnet)
            err_evolution = [float(e[1]) for e in result]
            output.append((subnet, err_evolution))
            for selected in nodes:
                if selected in pool: pool.remove(selected)
            print("nodes remaining: ", len(pool))
    print("all nodes selected")
    return output

def subnets_ni(channels_dict, labels_list, directed_pairs, nonselected_nodes):
    '''iterative version;
    selects functional subnetworks from a pool of nodes;
    each node can belong to multiple subnetworks;
    process ends when all nodes are selected'''
    seeds, pairs, pool = deepcopy(channels_dict), deepcopy(directed_pairs), deepcopy(nonselected_nodes)
    output = []
    while pool:
        if len(pool)==1:
            subnet = pool
            output.append(subnet)
            print("1 unassigned node: ", pool[0])
            pool.pop()
            return output
        else:
            print("starting a new subnetwork... ")
            n, connectivity, nodes, result = 2, {}, [], []
            subset = list(seeds.keys()) #continue with all nodes
            while n<len(subset)+1:
                if n==2: connectivity = pairs
                elif n>2: connectivity = PEC_multi(subset, n, seeds, nodes)
                result.append(min(connectivity.items(), key=lambda x: x[1]))
                errors = list(tup[1] for tup in result)
                mav_errors = moving_average(errors, 2)
                if len(mav_errors)>1 and mav_errors[-1]>mav_errors[-2]: 
                    print("moving average increased ", mav_errors[-2], "<", mav_errors[-1])
                    del pairs[result[0][0]]
                    result.pop(-1)
                    nodes=result[-1][0]
                    n=len(subset)+1
                else: 
                    nodes=result[-1][0]
                    n+=1
                    print("adding nodes...")
            subnet = [labels_list[nd] for nd in nodes]
            print(result, subnet)
            err_evolution = [float(e[1]) for e in result]
            output.append((subnet, err_evolution))
            for selected in nodes:
                if selected in pool: pool.remove(selected)
            print("nodes remaining: ", len(pool))
    print("all nodes selected")
    return output

def multipec(channels_dict, labels_list, pairs, nonselected_nodes, limit=False, n_nets=10):
    seeds, pairs, pool = deepcopy(channels_dict), deepcopy(pairs), deepcopy(nonselected_nodes)
    remaining_pairs = list(pairs.keys())
    print(remaining_pairs)
    output = []
    for i in range(n_nets):

          print("starting a new subnetwork... ")
          n, connectivity, nodes, result = 2, {}, [], []
          subset = deepcopy(pool)

          n_max = len(subset)
          if type(limit)==int:
              n_max = limit if limit<len(subset) else len(subset)
              print(f'network size limit = {n_max}')

          while n<=n_max:
              if n==2: connectivity = {i:pairs[i] for i in pairs if i in remaining_pairs}
              elif n>2: connectivity = PEC_multi(subset, n, seeds, nodes)
              result.append(min(connectivity.items(), key=lambda x: x[1]))
              errors = list(tup[1] for tup in result)
              if len(errors)>1 and errors[-1]>errors[-2]:
                  print("moving average increased ", errors[-2], "<", errors[-1])
                  del result[-1]
                  nodes=result[-1][0]
                  n=n_max+1
              else:
                  nodes=result[-1][0]
                  n+=1
                  print("adding nodes...")

          subnet = nodes
          print(result, subnet)
          if len(result)==0: break
          err_evolution = [float(e[1]) for e in result]
          output.append((subnet, err_evolution))
          if (nodes[0],nodes[1]) in remaining_pairs: remaining_pairs.remove((nodes[0],nodes[1]))

    if output[-1]==[]: output.pop()
    print(f"{n_nets} iterations finished")
    return output