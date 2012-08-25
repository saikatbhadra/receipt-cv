#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  get_receipt_data.py
#
import numpy as np
import re
import decimal
from math import *

def split_receipt(rlist):

    #TODO: This is also written badly

    if len(rlist) < 3:
        return (None,None)
    
    #iterate through numbers to find th
    totals_idx = [None,None]
    totals_max = [None,None]

    iterator = 0
    for quant, desc, value in rlist:
        if value > totals_max[1]:
            totals_max[0] = totals_max[1]
            totals_max[1] = value
            totals_idx[0] = totals_idx[1]
            totals_idx[1] = iterator
        iterator += 1

    #check sum
    subtotal_sum = 0
    for e in range(totals_idx[0]):
        subtotal_sum += rlist[e][2]

    total_sum = 0
    for e in range(totals_idx[0],totals_idx[1]):
        total_sum += rlist[e][2]

    items = []
    subtotal = None
    charges = []
    total = None

    if (subtotal_sum != totals_max[0]) or (total_sum != totals_max[1]):
        return items, substotal, charges, total

    subtotal = rlist[totals_idx[0]][2]
    total = rlist[totals_idx[1]][2]
   

    for item_idx in range(len(rlist)):
        if item_idx < totals_idx[0]:
            items.append(rlist[item_idx])
        elif (item_idx < totals_idx[1]) and (item_idx > totals_idx[0]):
            charges.append(rlist[item_idx])

    return items, subtotal, charges, total    

def fix_ocr(lines):
    #TODO: This is written terribly, please fix
    fix = False

    # fix all 7s that are ?s
    # todo check that the fixed string is a dollar amount   
    seven_exp = re.compile('\d\?')
    seven_exp2 = re.compile('\?\d')
    zero_exp = re.compile('[a-zA-Z]0[a-zA-Z]')
    one_exp = re.compile('[a-zA-Z]1[a-zA-Z]')
    line_num = 0
    
    for line in lines:
        # fix ? marks
        exp = re.search(seven_exp,line)
        if exp:
            idx = int(exp.end())-1
            line = line[:idx] + '7' + line[idx+1:]
            fix = True

        exp = re.search(seven_exp2,line)
        if exp:
            idx = int(exp.start())
            line = line[:idx] + '7' + line[idx+1:]
            fix = True
            
        # fix 0s
        exp = re.search(zero_exp,line)
        if exp:
            idx = int(exp.start())+1
            line = line[:idx] + 'o' + line[idx+1:]
            fix = True

        """# fix 1s
        exp = re.search(one_exp,line)
        if exp:
            idx = int(exp.start())+1
            line = line[:idx] + 'i/l' + line[idx+1:]
            fix = True

        """      
            
        lines[line_num] = line
        line_num += 1

    return fix
        

def extract_items(lines):
    item_exp = re.compile('\d*\d\.\d\d')
    
    item_lines = []
    list_features = []
    for line in lines:
        cost = re.search(item_exp,line)
        if cost:
            desc = re.sub(item_exp,'',line)
           
            quantity_exp = re.match('\d*\d',line)
            if quantity_exp:
                quantity = int(quantity_exp.group())
                # remove quantity from descriptions
                desc = re.sub(quantity_exp.group(),'',desc)
            else:
                quantity = 1

            desc = desc.strip() #remove white space and /n
            desc = desc.capitalize()
            value = decimal.Decimal(cost.group())
            list_features.append((quantity, desc, value))
    return list_features
        
def get_receipt_data(txt_loc):
    
    #read in OCR output
    txt_file = open(txt_loc, 'r')
    lines = txt_file.readlines()
    txt_file.close()

    # attempt badly to try to fix items
    while fix_ocr(lines):
        pass

    # extract out items, charges, totals
    price_items = extract_items(lines)
    items, subtotal, charges, total = split_receipt(price_items)
    return [items, subtotal, charges, total]
	
items = get_receipt_data('taco.txt')
	


def main():
	
	return 0

if __name__ == '__main__':
	main()

