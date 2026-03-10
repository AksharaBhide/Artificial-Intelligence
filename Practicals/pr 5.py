print("CRYPTARITHMETIC PUZZLE SOLVER")
print("=" * 60)
print("Solving: SEND + MORE = MONEY")
print("=" * 60)

# Counter for solutions found
solution_count = 0
all_solutions = []

print("\nSearching for ALL possible solutions...")
print("(This might take a few seconds)\n")

# Try ALL possible combinations for all 8 letters
for S in range(1, 10): # S cannot be 0
    for E in range(0, 10):
        for N in range(0, 10):
            for D in range(0, 10):
                for M in range(1, 10): # M cannot be 0
                    for O in range(0, 10):
                        for R in range(0, 10):
                            for Y in range(0, 10):
                                # Check that all 8 digits are different
                                digits = [S, E, N, D, M, O, R, Y]
                                if len(set(digits)) != 8:
                                    continue
                                
                                # Calculate the numbers
                                SEND = S*1000 + E*100 + N*10 + D
                                MORE = M*1000 + O*100 + R*10 + E
                                MONEY = M*10000 + O*1000 + N*100 + E*10 + Y
                                
                                # Check if SEND + MORE = MONEY
                                if SEND + MORE == MONEY:
                                    solution_count += 1
                                    all_solutions.append({
                                        'S': S, 'E': E, 'N': N, 'D': D,
                                        'M': M, 'O': O, 'R': R, 'Y': Y,
                                        'SEND': SEND, 'MORE': MORE, 'MONEY': MONEY
                                    })

# Display results
if solution_count > 0:
    print(f"\n🎉 FOUND {solution_count} SOLUTION(S)!")
    print("=" * 60)
    
    for i, sol in enumerate(all_solutions, 1):
        print(f"\nSOLUTION {i}:")
        print("-" * 40)
        
        # Show alphabet values
        print("ALPHABET VALUES:")
        print(f" S = {sol['S']}")
        print(f" E = {sol['E']}")
        print(f" N = {sol['N']}")
        print(f" D = {sol['D']}")
        print(f" M = {sol['M']}")
        print(f" O = {sol['O']}")
        print(f" R = {sol['R']}")
        print(f" Y = {sol['Y']}")
        
        # Show the numbers
        print("\nNUMBERS:")
        print(f" SEND = {sol['S']}{sol['E']}{sol['N']}{sol['D']} = {sol['SEND']}")
        print(f" MORE = {sol['M']}{sol['O']}{sol['R']}{sol['E']} = {sol['MORE']}")
        print(f" MONEY = {sol['M']}{sol['O']}{sol['N']}{sol['E']}{sol['Y']} = {sol['MONEY']}")
        
        # Verify the equation
        print(f"\nVERIFICATION:")
        print(f" {sol['SEND']} + {sol['MORE']} = {sol['MONEY']}")
        print(f" → {sol['SEND'] + sol['MORE']} = {sol['MONEY']}")
        
        print("-" * 40)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Total unique solutions found: {solution_count}")
    if solution_count == 1:
        print("This puzzle has only ONE unique solution!")
else:
    print("\n❌ No solutions found!")

print("=" * 60)