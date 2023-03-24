import pandas
import copy

percent = lambda x : x/100

def format_number(num):
  string = '{:,.2f}'.format(num)

  return string

def df_to_dict(df, Skus):

  Demand = {}

  for i, s in enumerate (Skus):

    row = df.iloc[i]
    row = row.to_dict()
    row = {int(k):int(v) for k,v in row.items()}

    sku_demand = {s: row} 
    #print(sku_demand)

    #create a new dictionary {A:{row A demand},... D:{row D demand}}
    Demand = dict(**Demand, **sku_demand)

  return Demand

def calculate_total_demand(Demand, Skus, Periods):
  
  total_demand = 0

  for i in Skus:
      sku_demand = Demand[i]
      for j in Periods:
        total_demand += sku_demand[j]

  return total_demand

def calculate_future_demand(Demand, Skus, Periods, compound_growth, period_year=0):

  future_demand = 0
  Demand_future = copy.deepcopy(Demand)

  for i in Skus:
    sku_demand = Demand_future[i]

    for j in Periods:
      future_demand = sku_demand[j]*(1+compound_growth)**period_year
      Demand_future[i][j] = future_demand

  return Demand_future


def calculate_future_demand2(Demand, Skus, Periods, growth, period_year=0):

  future_demand = 0
  Demand_future = copy.deepcopy(Demand)

  for i in Skus:
    sku_demand = Demand_future[i]

    for j in Periods:
      future_demand = sku_demand[j]*(1+percent(growth[i][j]))**period_year
      Demand_future[i][j] = future_demand

  return Demand_future


def calculate_total_production(Shifts, hours_shift, Rates):

  output = 0
  counter = 0

  total_shifts = 0
  rate_counter = 0

  for s in Shifts.values():

    output += s.varValue * list(Rates.values())[rate_counter]*hours_shift
    #print(list(Rates.values())[rate_counter], s.varValue, counter)

    total_shifts += s.varValue
    if (counter == 11):
      rate_counter +=1
      counter = -1
    
    counter +=1

  return output

def calculate_total_production2(Shifts, hours_shift, Rates):

  output = 0
  counter = 0

  total_shifts = 0
  rate_counter = 0

  for s in Shifts.values():
  
    output += s.varValue * list(Rates.values())[rate_counter][counter+1]*hours_shift
    
    total_shifts += s.varValue
    if (counter == 11):
      rate_counter +=1
      counter = -1
    
    counter +=1

  return output


def find_output_list(Shifts, Rates, Skus, hours_shift):
  
  output = 0
  counter = 0

  total_shifts = 0
  rate_counter = 0

  production_output = []

  for s in Shifts.values():

    output += s.varValue * list(Rates.values())[rate_counter]*hours_shift
    #print(list(Rates.values())[rate_counter], s.varValue, counter)

    production_output.append((Skus[rate_counter], counter+1, s.varValue * list(Rates.values())[rate_counter]*hours_shift))

    total_shifts += s.varValue
    if (counter == 11):
      rate_counter +=1
      counter = -1
    
    counter +=1

  return production_output

def find_output_list2(Shifts, Rates, Skus, hours_shift):
  
  output = 0
  counter = 0

  total_shifts = 0
  rate_counter = 0

  production_output = []

  for s in Shifts.values():

    output += s.varValue * list(Rates.values())[rate_counter][counter+1]*hours_shift
    #print(list(Rates.values())[rate_counter], s.varValue, counter)

    production_output.append((Skus[rate_counter], counter+1, s.varValue * list(Rates.values())[rate_counter][counter+1]*hours_shift))

    total_shifts += s.varValue
    if (counter == 11):
      rate_counter +=1
      counter = -1
    
    counter +=1

  return production_output

def convert_output_list_to_dict(output_list, Skus):
  
  counter = 0
  sku_counter = 0
  
  temp_list = []
  temp_dict ={}
  outputDictionary ={}

  for item in output_list:

    if counter == 11:

      temp_list.append((item[1], item[2]))
      counter = 0
      temp_dict = dict(temp_list)
      #print(Skus[sku_counter], temp_dict)

      output_vector = {Skus[sku_counter]: copy.deepcopy(temp_dict)} 
      outputDictionary = dict(**outputDictionary, **output_vector)
      
      sku_counter +=1
      temp_list.clear()
      temp_dict.clear()
    else:
      
      temp_list.append((item[1], item[2]))
      counter += 1

  return outputDictionary


def find_diff(outputDictionary, Demand, Skus, Periods): 
 
  temp_dict = {}
  temp_list = []
  diffDictionary = {}

  for s in Skus:
    #print(Demand[s])
    #print(outputDictionary[s],'\n')

    for j in Periods:
      diff = int(outputDictionary[s][j] - Demand[s][j])
      temp_list.append((j,diff))

    temp_dict = dict(temp_list)
    
    output_vector = {s:copy.deepcopy(temp_dict)}
    diffDictionary = dict(**diffDictionary, **output_vector)

  return diffDictionary


