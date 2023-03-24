from pulp import *
from demand_module import *
import pandas as pd


def analyze(Skus, Periods, Rates, hours_shift, Max_shifts, Demand):
    #ANALYSIS

    prob = LpProblem("Brampton_Capacity", LpMaximize)

    #Variables

    Shifts = LpVariable.dicts("Shifts", [(i,j) for i in Skus for j in Periods], 0)
    max_shifts = LpVariable("Max", 0)
    total_production = LpVariable("Production",0)

    #Variables definitions
    total_production += lpSum(Shifts[i,j]*12*Rates[i][j] for j in Periods for i in Skus)
    max_shifts += lpSum(Shifts[i,j] for j in Periods for i in Skus)

    #Objective
    prob += max_shifts

    #Constraints

    #Contraint 1 :Restrict Output April and May shifts = 0


    #Constraint 2: v_ij <= d_ij
    for i in Skus:

      sku_demand = Demand[i]
      for j in Periods:

        inv = Rates[i][j]*hours_shift    
        prob += LpConstraint(Shifts[i,j]*inv <= sku_demand[j])

    #Constraing 3: max_shifts <= sum(Max_shifts)
    prob += LpConstraint(max_shifts <= sum(Max_shifts))

    #Solve and evaluate results
    prob.solve()

    return prob, Shifts

def calculate_spare_SPCs(Shifts, Max_shifts, hours_shift, t):

    total = 0

    for s in Shifts.values():
        total += s.varValue

    #SPC 50% of 12 pack rate
    return (int(sum(Max_shifts) - total)*hours_shift*t*0.5)    

def project_20years(Demand, Skus, Periods, growth, Rates, t, hours_shift, Max_shifts, year0=2023):

    delta_by_year = []

    #20 years projection with given Rates
    for year in range(0,20,1):
        future_demand = calculate_future_demand2(Demand, Skus, Periods, growth, period_year=year)

        prob, Shifts = analyze(Skus, Periods, Rates, hours_shift, Max_shifts, future_demand)
        total_demand = calculate_total_demand(future_demand, Skus, Periods)
        #print(format_number(total_demand))

   
        output = calculate_total_production2(Shifts, hours_shift, Rates)
        delta = int(output - total_demand)

        if delta < 0:
            delta_by_year.append((str(year + year0), delta))
        else:
            delta_by_year.append((str(year + year0), calculate_spare_SPCs(Shifts, Max_shifts, hours_shift, t)))

    return(dict(delta_by_year))


def test_rates_20years(test_rates, initial_rate, Max_rate, Rates, Skus, Periods, growth, hours_shift, Max_shifts, Demand):

    matrix = []

    for t in test_rates:

        Rates_test = copy.deepcopy(Rates)
        factor  = t/initial_rate

        for i in Skus:
            for j in Periods:
                Rates_test[i][j] = Rates[i][j]*factor

        rate_t = project_20years(Demand, Skus, Periods, growth, Rates_test, t, hours_shift, Max_shifts, year0 =2023)
        matrix.append(( (str(t/Max_rate*100) + '%'), rate_t))


    #print(dict(matrix))        
    matrix_dict = dict(matrix)

    return matrix_dict


def _color_red_or_green(val):

    is_red = val < 0 
    return['background-color: red' if val else 'background-color: green' for val in is_red]


def main():
    #LOAD DATA

    df = pd.read_csv("BR1_Y0.csv", index_col=0)

    print(df)
    #df.style

    #KEY INPUTS

    Skus = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
    Periods = [1,2,3,4,5,6,7,8,9,10,11,12]

    Max_shifts = [41,41,52,46,46,58,43,43,54,43,43,54]
    hours_shift = 12


    #To restrict the output set rate to zero, for example April 0, 4:0

    Rates =  {'A': {1:7000, 2:7000, 3:7000, 4:7000, 5:7000, 6:7000, 7:7000, 8:7000, 9:7000, 10:7000, 11:7000, 12:7000},
              'B': {1:3500, 2:3500, 3:3500, 4:3500, 5:3500, 6:3500, 7:3500, 8:3500, 9:3500, 10:3500, 11:3500, 12:3500},
              'C': {1:4667, 2:4667, 3:4667, 4:4667, 5:4667, 6:4667, 7:4667, 8:4667, 9:4667, 10:4667, 11:4667, 12:4667},
              'D': {1:2625, 2:2625, 3:2625, 4:2625, 5:2625, 6:2625, 7:2625, 8:2625, 9:2625, 10:2625, 11:2625, 12:2625},
              'E': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'F': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'G': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'H': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'I': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'J': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'K': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'L': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'M': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
              'N': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
             }


    Demand = df_to_dict(df, Skus)
    print("Demand in dictionary format: ", Demand, '\n')

    total_demand = calculate_total_demand(Demand, Skus, Periods)
    print(format_number(total_demand), '\n')

    
    #ANALYZE
    prob, Shifts = analyze(Skus, Periods, Rates, hours_shift, Max_shifts, Demand)

    #Review Solution
    for v in prob.variables():
      print(v.name, "=", v.varValue)

    #Total Demand vs Total Production
    total_demand = calculate_total_demand(Demand, Skus, Periods)
    print(format_number(total_demand))

    output = calculate_total_production2(Shifts, hours_shift, Rates)
    print(format_number(output))

    print ("Delta: Production vs Demand:", format_number(output - total_demand))

    #Find V and v_ij
    output_list = find_output_list2(Shifts, Rates, Skus, hours_shift)
    outputDictionary = convert_output_list_to_dict(output_list, Skus)
    #print(outputDictionary)
    
    diffDictionary = find_diff(outputDictionary, Demand, Skus, Periods)
    print(diffDictionary)

    #Future Demand
    compound_growth = percent(1.1)
    future_demand = calculate_future_demand(Demand, Skus, Periods, compound_growth, period_year=1)
    print(format_number(calculate_total_demand(future_demand, Skus, Periods)))
    
    growth = {
           'A': {1:1.1,	2:1.1,	3:1.1,	4:1.1,	5:1.1,	6:1.1,	7:1.1,	8:1.1,	9:1.1,	10:1.1,	11:1.1,	12:1.1},
           'B': {1:1.1,	2:1.1,	3:1.1,	4:1.1,	5:1.1,	6:1.1,	7:1.1,	8:1.1,	9:1.1,	10:1.1,	11:1.1,	12:1.1},
           'C': {1:1.1,	2:1.1,	3:1.1,	4:1.1,	5:1.1,	6:1.1,	7:1.1,	8:1.1,	9:1.1,	10:1.1,	11:1.1,	12:1.1},
           'D': {1:1.1,	2:1.1,	3:1.1,	4:1.1,	5:1.1,	6:1.1,	7:1.1,	8:1.1,	9:1.1,	10:1.1,	11:1.1,	12:1.1},
           'E': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'F': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'G': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'H': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'I': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'J': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'K': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'L': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'M': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0},
           'N': {1:0, 2:0, 3:0,	4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}
               
          }
    future_demand2 = calculate_future_demand2(Demand, Skus, Periods, growth, period_year=1)
    print(format_number(calculate_total_demand(future_demand2, Skus, Periods)))

    delta_demand = find_diff(future_demand, Demand, Skus, Periods)
    print(format_number(calculate_total_demand(delta_demand, Skus, Periods)))
    


if __name__ == '__main__':

    main()


    
