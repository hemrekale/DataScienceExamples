# [source]: https://realpython.com/linear-programming-python/#linear-programming-examples



# SciPy disadvantages in solving linear programming

# 1 SciPy can’t run various external solvers.
# 2 SciPy can’t work with integer decision variables.
# 3 SciPy doesn’t provide classes or functions that facilitate model building. You have to define arrays and matrices, which might be a tedious and error-prone task for large problems.
# 4 SciPy doesn’t allow you to define maximization problems directly. You must convert them to minimization problems.
# 5 SciPy doesn’t allow you to define constraints using the greater-than-or-equal-to sign directly. You must use the less-than-or-equal-to instead.


# Example1
from scipy.optimize import linprog

obj = [-1, -2]
#      ─┬  ─┬
#       │   └┤ Coefficient for y
#       └────┤ Coefficient for x

lhs_ineq = [[ 2,  1],  # Red constraint left side
            [-4,  5],  # Blue constraint left side
            [ 1, -2]]  # Yellow constraint left side

rhs_ineq = [20,  # Red constraint right side
            10,  # Blue constraint right side
             2]  # Yellow constraint right side

lhs_eq = [[-1, 5]]  # Green constraint left side
rhs_eq = [15]       # Green constraint right side

bnd = [(0, float("inf")),  # Bounds of x
       (0, float("inf"))]  # Bounds of y

opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
              A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
              method="revised simplex")
opt


# Now an example with PuLP
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
# Create the model
model = LpProblem(name="small-problem", sense=LpMaximize)

# Initialize the decision variables
x = LpVariable(name="x", lowBound=0, cat="Integer")
y = LpVariable(name="y", lowBound=0)

# Add the constraints to the model
model += (2 * x + y <= 20, "red_constraint")
model += (4 * x - 5 * y >= -10, "blue_constraint")
model += (-x + 2 * y >= -2, "yellow_constraint")
model += (-x + 5 * y == 15, "green_constraint")

# Add the objective function to the model
obj_func = x + 2 * y
model += obj_func

# Add the objective function to the model
model += x + 2 * y

#or for larger models 
# Add the objective function to the model
model += lpSum([x, 2 * y])

# Solve the problem
status = model.solve()

# print results

print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")


for var in model.variables():
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")

model.variables()
model.variables()[0]
model.variables()[1]

# Example 2 Profit Example
# SciPy
obj = [-20, -12, -40, -25]

lhs_ineq = [[1, 1, 1, 1],  # Manpower
            [3, 2, 1, 0],  # Material A
            [0, 1, 2, 3]]  # Material B

rhs_ineq = [ 50,  # Manpower
            100,  # Material A
             90]  # Material B

opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
              method="revised simplex")
opt


# PuLP solution to example 2

# Define the model
model = LpProblem(name="resource-allocation", sense=LpMaximize)

# Define the decision variables
x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 5)}

# Add constraints
model += (lpSum(x.values()) <= 50, "manpower")
model += (3 * x[1] + 2 * x[2] + x[3] <= 100, "material_a")
model += (x[2] + 2 * x[3] + 3 * x[4] <= 90, "material_b")

# Set the objective
model += 20 * x[1] + 12 * x[2] + 40 * x[3] + 25 * x[4]

# Solve the optimization problem
status = model.solve()

# Get the results
print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")

for var in x.values():
    print(f"{var.name}: {var.value()}")

for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")


# With GLPK

from pulp import GLPK

# Create the model
model = LpProblem(name="small-problem", sense=LpMaximize)

# Initialize the decision variables
x = LpVariable(name="x", lowBound=0)
y = LpVariable(name="y", lowBound=0)

# Add the constraints to the model
model += (2 * x + y <= 20, "red_constraint")
model += (4 * x - 5 * y >= -10, "blue_constraint")
model += (-x + 2 * y >= -2, "yellow_constraint")
model += (-x + 5 * y == 15, "green_constraint")

# Add the objective function to the model
model += lpSum([x, 2 * y])

# Solve the problem
status = model.solve(solver=GLPK(msg=False))

print(f"status: {model.status}, {LpStatus[model.status]}")


print(f"objective: {model.objective.value()}")


for var in model.variables():
    print(f"{var.name}: {var.value()}")


for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")


# Add a constraint to the model
# For the Profit example now 1st and 3rd product cannot be produced at once.

model = LpProblem(name="resource-allocation", sense=LpMaximize)


# Define the decision variables

x = {i: LpVariable(name=f"x{i}", lowBound=0) for i in range(1, 5)}
y = {i: LpVariable(name=f"y{i}", cat="Binary") for i in (1, 3)}


# Add constraints

model += (lpSum(x.values()) <= 50, "manpower")
model += (3 * x[1] + 2 * x[2] + x[3] <= 100, "material_a")
model += (x[2] + 2 * x[3] + 3 * x[4] <= 90, "material_b")


M = 100

model += (x[1] <= y[1] * M, "x1_constraint")
model += (x[3] <= y[3] * M, "x3_constraint")
model += (y[1] + y[3] <= 1, "y_constraint")

# Set objective
model += 20 * x[1] + 12 * x[2] + 40 * x[3] + 25 * x[4]

# Solve the optimization problem

status = model.solve()

print(f"status: {model.status}, {LpStatus[model.status]}")
print(f"objective: {model.objective.value()}")

for var in model.variables():
    print(f"{var.name}: {var.value()}")


for name, constraint in model.constraints.items():
    print(f"{name}: {constraint.value()}")

# PULP tutorial II
# [source][2] :https://towardsdatascience.com/linear-programming-and-discrete-optimization-with-python-using-pulp-449f3c5f6e99

import pandas as pd
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, value


df = pd.read_excel("data/diet_medium.xls",nrows = 17)
df.head()
df.info()

df = df[~df['Foods'].str.contains("Eggs")]
df = df[~df['Foods'].str.contains("Cookies")]
df = df[~df['Foods'].str.contains("Beef")]
df = df[~df['Foods'].str.contains("Chicken")]
df = df[~df['Foods'].str.contains("Turkey")]

#Create the PuLP problem variable. Since it is a cost minimization problem, we need to use LpMinimize
# Create the 'prob' variable to contain the problem data
prob = LpProblem("Simple Diet Problem", LpMinimize)

# Creates a list of the Ingredients
food_items = list(df['Foods'])

print("So, the food items to consdier, are\n"+"-"*100)
for f in food_items:
    print(f,end=', ')

calories = dict(zip(food_items,df['Calories']))
cholesterol = dict(zip(food_items,df['Cholesterol (mg)']))
fat = dict(zip(food_items,df['Total_Fat (g)']))
sodium = dict(zip(food_items,df['Sodium (mg)']))
carbs = dict(zip(food_items,df['Carbohydrates (g)']))
fiber = dict(zip(food_items,df['Dietary_Fiber (g)']))
protein = dict(zip(food_items,df['Protein (g)']))
vit_C = dict(zip(food_items,df['Vit_C (IU)']))
calcium = dict(zip(food_items,df['Calcium (mg)']))
iron = dict(zip(food_items,df['Iron (mg)']))

# A dictionary called 'food_vars' is created to contain the referenced Variables
food_vars = LpVariable.dicts("Food",food_items,0,cat='Continuous')
# The objective function is added to 'prob' first
prob += lpSum([costs[i]*food_vars[i] for i in food_items]), "Total Cost of the balanced diet"

#Adding calorie constraint
prob += lpSum([calories[f] * food_vars[f] for f in food_items]) >= 800.0, "CalorieMinimum"
prob += lpSum([calories[f] * food_vars[f] for f in food_items]) <= 1300.0, "CalorieMaximum"

# Fat
prob += lpSum([fat[f] * food_vars[f] for f in food_items]) >= 20.0, "FatMinimum"
prob += lpSum([fat[f] * food_vars[f] for f in food_items]) <= 50.0, "FatMaximum"

# Carbs
prob += lpSum([carbs[f] * food_vars[f] for f in food_items]) >= 130.0, "CarbsMinimum"
prob += lpSum([carbs[f] * food_vars[f] for f in food_items]) <= 200.0, "CarbsMaximum"

# Fiber
prob += lpSum([fiber[f] * food_vars[f] for f in food_items]) >= 60.0, "FiberMinimum"
prob += lpSum([fiber[f] * food_vars[f] for f in food_items]) <= 125.0, "FiberMaximum"

# Protein
prob += lpSum([protein[f] * food_vars[f] for f in food_items]) >= 100.0, "ProteinMinimum"
prob += lpSum([protein[f] * food_vars[f] for f in food_items]) <= 150.0, "ProteinMaximum"

# The problem data is written to an .lp file
prob.writeLP("SimpleDietProblem.lp")
# The problem is solved using PuLP's choice of Solver
prob.solve()
# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])
print("Therefore, the optimal (least cost) balanced diet consists of\n"+"-"*110)
for v in prob.variables():
    if v.varValue>0:
        print(v.name, "=", v.varValue)

print("The total cost of this balanced diet is: ${}".format(round(value(prob.objective),2)))

prob.objective

obj = value(prob.objective) 
print("The total cost of this balanced diet is: ${}".format(round(obj,2)))


